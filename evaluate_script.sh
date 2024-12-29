#!/bin/bash

# Execute Python script
python ./evaluation/evaluate_model_lora.py \
  --model_name models/whisper-large-v2-sr-lora \
	--language Serbian \
	--config sr \
	--save_transcript \
	--output_file eval-large-v2-sr-lora-500-eval-steps.txt \
	--dataset_name ParlaSpeechSR+Fleurs+CommonVoice \
	--cer \
	--ref_key transcription
