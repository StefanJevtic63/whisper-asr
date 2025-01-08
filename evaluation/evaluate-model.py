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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperProcessor

from torch.utils.data import DataLoader
from tqdm import tqdm

# add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ASR')))
from data.train_data.DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding
from data.evaluation_data.SerbianCyrillicNormalizer import SerbianCyrillicNormalizer
from data.evaluation_data.SpellChecker import SpellChecker
from data.evaluation_data.WordFrequencies import WordFrequencies

def save_results(references, predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'references': references, 'predictions': predictions}, f, ensure_ascii=False, indent=4)

def load_results(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        references = data['references']
        predictions = data['predictions']

        return references, predictions

def evaluate_model(references, predictions, args, output_dir, is_spell_check):
    # load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    wer = wer_metric.compute(references=references, predictions=predictions)

    # determine whether to calculate additional metrics
    if args.cer:
        cer = cer_metric.compute(references=references, predictions=predictions)

    os.makedirs(output_dir, exist_ok=True)

    # output results to a file
    output_file = args.output_file
    if is_spell_check:
        output_file = output_file.removesuffix(".txt")
        output_file += "-spell-check.txt"

    output_file = os.path.join(output_dir, output_file)

    with open(output_file, 'w', encoding="utf-8") as file:
        file.write(f"[INFO] Evaluated {args.model_name} with {args.dataset_name} "
                   f"with language {args.language}/{args.config}\n\n")
        file.write(f"WER : {round(100 * wer, 4)}\n\n")

        # determine whether to print additional metrics
        if args.cer:
            file.write(f"CER : {round(100 * cer, 4)}\n\n")

        if args.save_transcript:
            for ref, pred in zip(references, predictions):
                file.write(f"Reference: {ref}\nPrediction: {pred}\n{'-' * 40}\n")

    # testing finished
    print(f"[INFO] Testing finished and model was evaluated at {output_file}")


# change constants accordingly
TASK = "transcribe"
CURRENT_DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIRECTORY_PATH, "../data/datasets/test_dataset_v2")
DICTIONARY_PATH = os.path.join(CURRENT_DIRECTORY_PATH, "../data/evaluation_data/serbian-cyrillic.dic")
SERIALIZE_OUTPUT_FILE = os.path.join(CURRENT_DIRECTORY_PATH, "result-data-whisper-v2.txt")
OUTPUT_DIR = os.path.join(CURRENT_DIRECTORY_PATH, "evaluation_results")
BATCH_SIZE = 16

def main(args):
    print(
        f"[INFO] Evaluting {args.model_name} with {args.dataset_name} with language {args.language}/{args.config}"
    )

    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    # load processor
    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        language=args.language,
        task=TASK
    )

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
    predictions = [pred for pred, ref in zip(predictions, references) if ref.strip()]
    references = [ref for ref in references if ref.strip()]

    # save the results so they aren't lost if there's an error while calculating wer
    save_results(
        references=references,
        predictions=predictions,
        output_file=SERIALIZE_OUTPUT_FILE
    )

    print(f"[INFO] References and predictions saved to {SERIALIZE_OUTPUT_FILE}")

    # perform spell checking
    word_frequencies = WordFrequencies().calculate()
    spell_checker = SpellChecker(
        predictions=predictions,
        word_frequencies=word_frequencies,
        dictionary_path=DICTIONARY_PATH
    )
    predictions_spell_check = spell_checker.spell_check()

    # evaluate the predictions before and after spell checking
    evaluate_model(
        references=references,
        predictions=predictions,
        args=args,
        output_dir=OUTPUT_DIR,
        is_spell_check=False
    )
    evaluate_model(
        references=references,
        predictions=predictions_spell_check,
        args=args,
        output_dir=OUTPUT_DIR,
        is_spell_check=True
    )

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

    #################################################################################################

    references, predictions = load_results(SERIALIZE_OUTPUT_FILE)

    # perform spell checking
    word_frequencies = WordFrequencies().calculate()
    spell_checker = SpellChecker(
        predictions=predictions,
        word_frequencies=word_frequencies,
        dictionary_path=DICTIONARY_PATH
    )
    predictions_spell_check = spell_checker.spell_check()

    # evaluate the predictions before and after spell checking
    evaluate_model(
        references=references,
        predictions=predictions,
        args=args,
        output_dir=OUTPUT_DIR,
        is_spell_check=False
    )
    evaluate_model(
        references=references,
        predictions=predictions_spell_check,
        args=args,
        output_dir=OUTPUT_DIR,
        is_spell_check=True
    )

    #################################################################################################

    #main(args)