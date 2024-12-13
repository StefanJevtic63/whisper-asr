import argparse
import evaluate
import os
import sys
import numpy as np
import gc
import torch
import json

from peft import PeftModel, PeftConfig
from datasets import load_from_disk
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from SerbianCyrillicNormalizer import SerbianCyrillicNormalizer

# add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ASR')))
from DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding

def save_results(references, predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'references': references, 'predictions': predictions}, f, ensure_ascii=False, indent=4)

def load_results(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        references = data['references']
        predictions = data['predictions']

        return references, predictions


# change constants accordingly
TASK = "transcribe"
DATASET_PATH = "./datasets/test_dataset"
SERIALIZE_OUTPUT_FILE = "evaluation/result_data.txt"
OUTPUT_DIR = "./evaluation/evaluation-results"
BATCH_SIZE = 16

def main(args):
    print(
        f"[INFO] Evaluting {args.model_name} with {args.dataset_name} with language {args.language}/{args.config}"
    )

    # load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # load processor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v2", language=args.language, task=TASK)

    # load data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # load the dataset
    print(f"[INFO] Preparing data for testing phase...")
    dataset = load_from_disk(dataset_path=DATASET_PATH)

    print("[INFO] Structure of the loaded data:")
    print(dataset)

    dataset = dataset["test"]

    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=TASK)
    normalizer = SerbianCyrillicNormalizer()

    # load model
    peft_config = PeftConfig.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.model_name)

    predictions = []
    references = []

    # start testing
    print("[INFO] Starting testing...: ")

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend([normalizer(pred) for pred in decoded_preds])
                references.extend([normalizer(label) for label in decoded_labels])

            del generated_tokens, labels, batch
            gc.collect()

    # filter out any empty references
    filtered_predictions = [pred for pred, ref in zip(predictions, references) if ref.strip()]
    filtered_references = [ref for ref in references if ref.strip()]

    # save the results if there's an error while calculating wer, so the results aren't lost
    save_results(references=filtered_references,
                 predictions=filtered_predictions, 
                 output_file=SERIALIZE_OUTPUT_FILE)

    print(f"[INFO] References and prefictions saved to {SERIALIZE_OUTPUT_FILE}")

    # evaluate
    wer = wer_metric.compute(predictions=filtered_predictions, references=filtered_references)

    # determine whether to calculate additional metrics
    if args.cer:
        cer = cer_metric.compute(references=filtered_references, predictions=filtered_predictions)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # output results to a file
    output_file = os.path.join(OUTPUT_DIR, args.output_file)
    with open(output_file, 'w', encoding="utf-8") as file:
        file.write(f"[INFO] Evaluated {args.model_name} with {args.dataset_name} "
                   f"with language {args.language}/{args.config}\n\n")
        file.write(f"WER : {round(100 * wer, 4)}\n\n")

        # determine whether to print additional metrics
        if args.cer:
            file.write(f"CER : {round(100 * cer, 4)}\n\n")

        if args.save_transcript:
            for ref, pred in zip(filtered_references, filtered_predictions):
                file.write(f"Reference: {ref}\nPrediction: {pred}\n{'-' * 40}\n")

    # testing finished
    print(f"[INFO] Testing finished and model was evaluated at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper ASR Evaluation")

    parser.add_argument("--model_name", type=str, default="openai/whisper-small",
                        help="Hugging Face Whisper model name, or the path to a local model directory. (default: 'openai/whisper-small)")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_17_0",
                        help="Name of the dataset from Hugging Face (default: 'mozilla-foundation/common_voice_17_0')")
    parser.add_argument("--config", type=str, default="sr",
                        help="Configuration of the dataset (default: 'sr' for Serbian for Common Voice).")
    parser.add_argument("--language", type=str, default="sr",
                        help="Language code for transcription (default: 'sr' for Serbian for Common Voice)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU ID for using a GPU (e.g., 0), or -1 to use CPU. (default: 0)")
    parser.add_argument("--split", type=str, default="test",
                        help="The dataset split to evaluate on (default: 'test').")
    parser.add_argument("--output_file", type=str, default="whisper_eval",
                        help="Name of the file to save the evaluation results.")
    parser.add_argument("--ref_key", type=str, default="sentence",
                        help="Key in the dataset for reference data (default: 'sentence' - matches with Common Voice)")
    parser.add_argument("--save_transcript", action='store_true',
                        help="Flag to save the transcript to a file (default: False).")
    parser.add_argument("--cer", action='store_true',
                        help="Flag to calculate the Character Error Rate (default: False)")
    args = parser.parse_args()

    main(args)