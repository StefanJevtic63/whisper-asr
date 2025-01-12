import os
from datetime import datetime
import torch

from huggingface_hub import HfFolder
from datasets import load_from_disk
from transformers import (
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    WhisperFeatureExtractor, WhisperForConditionalGeneration,
    WhisperProcessor, WhisperTokenizer,
    BitsAndBytesConfig, EarlyStoppingCallback, get_scheduler
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from data.train_data.DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from data.train_data.SavePeftModelCallback import SavePeftModelCallback

# change constants as applicable
HF_API_KEY = ""
BASE_MODEL = "openai/whisper-large-v3"
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DIR_PATH, "data/datasets/train_validation_dataset_v3")
LOG_FILE_PATH = os.path.join(DIR_PATH, "log.txt")

# training constants
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
GENERATION_MAX_LENGTH = 128
LEARNING_RATE = 9e-4
WARMUP_STEPS = 400
MAX_STEPS = 16000
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 25
WEIGHT_DECAY = 1e-3
PATIENCE = 2


class WhisperASR:
    """Whisper Model for Automatic Speech Recognition (ASR) using Hugging Face's Transformers library."""

    def __init__(self, model_name="openai/whisper-small", dataset_name="mozilla-foundation/common_voice_17_0", existing_model=False, language="Serbian", language_code="sr", save_to_hf=False, output_dir="./models/whisper", ref_key="sentence"):
        """
        Initialize the model and load the data. 
        The default config is the small model trained on the Common Voice dataset for Serbian.

        :param str model_name: The model name from Hugging Face or custom path
            If 'existing_model' is True, this should be the path to the pre-trained model.
            Ex: "openai/whisper-small"

        :param bool existing_model: Flag to indicate whether to load an existing model from the specified
            'model_name' path. If False, a new model is initialized

        :param str language: The language of the model. Ex: "Serbian"
        :param str language_code: The language code of the model. Must match the language. Ex: "sr"
        :param str output_dir: The output directory of the model to save to
        :param bool save_to_hf: Whether to push to Hugging Face Repo
        :param str ref_key: The key to the reference data in the dataset
        """

        # setting up to save to hugging face repo
        self.save_to_hf = save_to_hf
        if save_to_hf:
            HfFolder.save_token(HF_API_KEY)  # token to save to HF

        self.dataset_name = dataset_name
        self.ref_key = ref_key

        # initialize model and tokenizer
        self.model_name = model_name
        self.language = language
        self.language_code = language_code
        self.existing_model = existing_model

        # initialize feature extractor, tokenizer and processor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_name, language=language, task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(self.model_name, language=language, task="transcribe")

        # check if cuda is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load correct model
        if existing_model:
            print(
                f"[INFO] Loading {self.model_name} model from existing model...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            self.model = self.model.to(device)
        else:
            print(
                f"[INFO] Loading {self.model_name} from hugging face library...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")

        # set the language explicitly to avoid incorrect language prediction
        self.model.generation_config.language = language.lower()
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id  # suppress warning
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        # post-processing by freezing all the model layers
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        # apply LORA
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"],
                                 lora_dropout=0.1, bias="none")

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # load data
        self._load_data()
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)
        self.output_dir = output_dir

        # this callback helps to save only the adapter weights and remove the base model weights
        self.save_peft_model_callback = SavePeftModelCallback()

    def _load_data(self):
        """Load the data from the Common Voice dataset and prepare it for training."""

        print(f"[INFO] Preparing data for training phase...")
        self.data = load_from_disk(dataset_path=DATASET_PATH)
        print("[INFO] Structure of the loaded data:")
        print(self.data)

    def _log(self, text):
        """
        Logs the given text to log.txt file.

        :param str text: The text to be logged
        """

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, "a+") as file:
            file.write(f"[{current_time}] " + text + "\n")

    def train(self):
        """
        Train the model for different values of hyperparameter grid_search.
        Set the training arguments using Seq2SeqTrainer.
        After training, save the model to the specified directory.
        """

        # configure training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="linear",
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            save_strategy="steps",
            predict_with_generate=True,
            generation_max_length=GENERATION_MAX_LENGTH,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            report_to=["tensorboard"],
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            push_to_hub=self.save_to_hf,
            save_safetensors=False,
            remove_unused_columns=False, # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
            label_names=["labels"],  # same reason as above
            dataloader_pin_memory=False,
        )

        # initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # initiate learning rate scheduler with weight decay
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS,
        )

        # initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["validation"],
            data_collator=self.data_collator,
            processing_class=self.processor.feature_extractor,

            # optimizer with weight decay and learning scheduler
            optimizers=(optimizer, lr_scheduler),

            # set callbacks
            callbacks=[self.save_peft_model_callback,
                       EarlyStoppingCallback(early_stopping_patience=PATIENCE),
                       ],
        )

        self.model.config.use_cache = False  # silence the warnings

        # start training
        print("[INFO] Starting training...: ")

        # resume training from the last checkpoint if it exists
        checkpoint_dirs = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint")]
        if checkpoint_dirs:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        # training finished and save model to model directory
        print(f"[INFO] Training finished and model saved to {self.output_dir}")

        # get the eval_loss for the current model
        eval_loss = trainer.state.best_metric
        self._log(f"Eval loss for model with learning_rate = {LEARNING_RATE} is {eval_loss}")

        # save model to hugging face
        if self.save_to_hf:
            kwargs = {
                "language": f"{self.language_code}",
                "model_name": f"Whisper - {self.language} Model",
                "finetuned_from": f"{BASE_MODEL}",
                "tasks": "automatic-speech-recognition",
            }

            trainer.push_to_hub(**kwargs)
            print(f"[INFO] Model saved to Hugging Face Hub")
