#!/bin/bash

# Activate conda environment
source /c/Users/Stefan/Documents/anaconda3/etc/profile.d/conda.sh
conda activate master-rad

# Execute Python script
python ./evaluation/evaluate_model_lora.py \
  --model_name models/whisper-medium-sr-lora \
	--language Serbian \
	--config sr \
	--save_transcript \
	--output_file eval-sr-medium-lora \
	--dataset_name mozilla-foundation/common_voice_17_0 \
	--cer \
	--ref_key sentence
