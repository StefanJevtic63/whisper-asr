#!/bin/bash

# Activate conda environment
source /c/Users/Stefan/Documents/anaconda3/etc/profile.d/conda.sh
conda activate master-rad

# Execute Python script
python train.py \
  --model_name openai/whisper-large-v2 \
  --language Serbian \
  --language_code sr \
  --output_dir ./models/whisper-large-v2-sr-lora \
  --save_to_hf \
  --ref_key transcription
