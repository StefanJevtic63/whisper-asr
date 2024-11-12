#!/bin/bash

# Activate conda environment
source /c/Users/Stefan/Documents/anaconda3/etc/profile.d/conda.sh
conda activate master-rad

# Execute Python script
python train.py \
  --model_name openai/whisper-small \
  --language Serbian \
  --language_code sr \
  --output_dir ./models/whisper-small-sr \
  --save_to_hf
