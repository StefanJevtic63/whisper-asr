#!/bin/bash

# Activate conda environment
source /c/Users/Stefan/Documents/anaconda3/etc/profile.d/conda.sh
conda activate master-rad

# Execute Python script
python ./evaluation/evaluate_model.py \
	--language sr \
	--config sr \
	--save_transcript \
	--output_file eval-sr-cv-standard \
	--dataset_name mozilla-foundation/common_voice_17_0 \
	--ref_key sentence
