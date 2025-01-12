#!/bin/bash

# Execute Python script
python ../evaluation/evaluate-model-lora.py \
  --model_name ../models/whisper-large-v3-sr-lora \
	--language Serbian \
	--config sr \
	--save_transcript \
	--output_file eval-large-v3-sr-lora.txt \
	--dataset_name ParlaSpeechSR+Fleurs+CommonVoice \
	--cer \
	--ref_key transcription
