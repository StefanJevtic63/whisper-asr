#!/bin/bash

# Execute Python script
python ../evaluation/evaluate-model.py \
  --model_name openai/whisper-large-v2 \
	--language Serbian \
	--config sr \
	--save_transcript \
	--output_file eval-large-v2-sr.txt \
	--dataset_name ParlaSpeechSR+Fleurs+CommonVoice \
	--cer \
	--ref_key transcription
