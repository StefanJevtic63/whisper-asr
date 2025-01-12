#!/bin/bash

# Execute Python script
python ../train.py \
  --model_name openai/whisper-large-v3 \
  --language Serbian \
  --language_code sr \
  --output_dir ../models/whisper-large-v3-sr-lora \
  --save_to_hf \
  --ref_key transcription
