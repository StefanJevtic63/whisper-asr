import os
from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import HfFolder
from transformers import (
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    WhisperFeatureExtractor, WhisperForConditionalGeneration,
    WhisperProcessor, WhisperTokenizer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from MetricsEval import MetricsEval
from SavePeftModelCallback import SavePeftModelCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# change constants as applicable
OUTPUT_DIR = "./models/whisper"
HF_API_KEY = ""
BASE_MODEL = "openai/whisper-large-v2"

# training constants
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-3
WARMUP_STEPS = 50
MAX_STEPS = 4000
EVAL_BATCH_SIZE = 8
SAVE_STEPS = 1000
EVAL_STEPS = 1000
LOGGING_STEPS = 25


class WhisperASR:
    """Whisper Model for Automatic Speech Recognition (ASR) using Hugging Face's Transformers library."""

    def __init__(self, model_name="openai/whisper-small", dataset_name="mozilla-foundation/common_voice_17_0", existing_model=False, language="Serbian", language_code="sr", save_to_hf=False, output_dir=OUTPUT_DIR, ref_key="sentence"):
        """
        Initialize the model and load the data. 
        The default config is the small model trained on the Common Voice dataset for Serbian.

        Args:
            model_name (str): The model name from Hugging Face or custom path
            If 'existing_model' is True, this should be the path to the pre-trained model. Ex: "openai/whisper-small"

            existing_model (bool): Flag to indicate whether to load an existing model from the specified 
            'model_name' path. If False, a new model is initialized

            language (str): The language of the model. Ex: "Serbian"
            language_code (str): The language code of the model. Must match the language. Ex: "sr"
            output_dir (str): The output directory of the model to save to
            save_to_hf (bool): Whether to push to Hugging Face Repo
            ref_key (str): The key to the reference data in the dataset
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

        self.train_split = "train+validation"
        self.test_split = "test"

        # initalize feature extractor, tokenizer and processor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            self.model_name, language=language, task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language=language, task="transcribe")

        # load correct model
        if existing_model:
            print(
                f"[INFO] Loading {self.model_name} model from existing model...")
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
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
                                 lora_dropout=0.05, bias="none")

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # load data
        self.data = DatasetDict()
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            self.processor)
        self._load_data()
        self.OUTPUT_DIR = output_dir

        # this callback helps to save only the adapter weights and remove the base model weights
        self.save_peft_model_callback = SavePeftModelCallback()

    def _load_data(self):
        """Load the data from the Common Voice dataset and prepare it for training."""

        print(
            f"[INFO] Preparing {self.dataset_name} data for training phase...")

        # load data from Common Voice dataset
        self.data["train"] = load_dataset(
            self.dataset_name, self.language_code, split=self.train_split, token=HF_API_KEY, trust_remote_code=True)
        self.data["test"] = load_dataset(
            self.dataset_name, self.language_code, split=self.test_split, token=HF_API_KEY, trust_remote_code=True)

        # only consider input audio and transcribed text to generalize dataset as much as possible
        self.data = self.data.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender",
             "locale", "path", "segment", "up_votes", "variant"]
        )

        print("[INFO] Structure of the loaded data:")
        print(self.data)

        print("[INFO] Sample entry from the training dataset: ")
        print(self.data["train"][0])

        # downsample audio data to 16kHz
        self.data = self.data.cast_column("audio", Audio(sampling_rate=16000))
        self.data = self.data.map(
            self._prepare_data, remove_columns=self.data.column_names["train"], num_proc=1)

    def _prepare_data(self, batch):
        """Converts audio files to the model's input feature format and encodes the target texts.

        Args:
            batch (dict): A batch of audio and text data.
        """

        # load and resample audio data from 48kHz to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch[self.ref_key]).input_ids
        return batch

    def train(self):
        """Train the model. Set the training arguments here and using Seq2SeqTrainer. 
        After training, save the model to the specified directory.
        """

        # metric evaluation for training
        eval_fn = MetricsEval(self.tokenizer)

        # configure training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.OUTPUT_DIR,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            max_steps=MAX_STEPS,
            gradient_checkpointing=True,
            fp16=True,
            num_train_epochs=1,
            eval_strategy="steps",
            save_strategy="steps",
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            predict_with_generate=True,
            generation_max_length=128,
            save_steps=SAVE_STEPS,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            #metric_for_best_model="wer", # because compute_metrics is not provided
            greater_is_better=False,
            push_to_hub=self.save_to_hf,
            save_safetensors=False,
            remove_unused_columns=False, # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
            label_names=["labels"],  # same reason as above
        )

        # initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.data["train"],
            eval_dataset=self.data["test"],
            data_collator=self.data_collator,
            # compute_metrics=eval_fn.compute, # removed because we cannot autocast
            processing_class=self.processor.feature_extractor,
            callbacks=[self.save_peft_model_callback],
        )

        self.model.config.use_cache = False  # silence the warnings
        self.processor.save_pretrained(training_args.output_dir)

        # start training
        print("[INFO] Starting training...: ")
        trainer.train()

        # training finished and save model to model directory
        print(f"[INFO] Training finished and model saved to {self.OUTPUT_DIR}")

        if self.save_to_hf:
            kwargs = {
                "language": f"{self.language_code}",
                "model_name": f"Whisper - {self.language} Model",
                "finetuned_from": f"{BASE_MODEL}",
                "tasks": "automatic-speech-recognition",
            }

            trainer.push_to_hub(**kwargs)
            print(f"[INFO] Model saved to Hugging Face Hub")
