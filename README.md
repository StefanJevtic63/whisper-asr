# ASR using OpenAI/Whisper

This repository contains the code and experiments for the Automatic Speech Recognition (ASR) project using the OpenAI/Whisper model. <br>
The project involves training, evaluating, and optimizing various Whisper model versions for ASR tasks.


## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [License](#license)


## Overview
The main goal of this project is to develop an efficient ASR system using the OpenAI/Whisper model. <br>
The project includes the following activities:
- Training and evaluating different versions of the Whisper model
- Utilizing techniques like parameter-efficient transfer learning (PEFT) and LoRA
- Implementing custom normalizer and language-specific configurations
- Performing grid search on hyperparameters such as learning rate and weight decay

### Datasets
Training, validation, and test splits from each dataset were merged to create a dataset of unified training, validation, and test splits. <br>

Datasets used:
- [ParlaSpeech-RS](https://huggingface.co/datasets/classla/ParlaSpeech-RS) <br>
- [Fleurs](https://huggingface.co/datasets/google/fleurs) <br>
- [Common Voice 17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) <br>

### Dictionary
The Serbian dictionary is available from the releases section. <br>
It was acquired from [turanjanin/spisak-srpskih-reci](https://github.com/turanjanin/spisak-srpskih-reci) repo. <br>

## Getting Started
### Cloning the repository
Clone the repository to a desired directory:
```bash
cd path/to/repo
git clone https://github.com/StefanJevtic63/whisper-asr.git
cd whisper-asr
```

### Prerequisites
If you're using Conda, install the required dependencies by creating a conda environment:

```bash
conda env create -f environment.yaml
conda activate whisper-asr
```

Otherwise, install the dependencies directly from the requirements.txt file:

```bash
pip install -r requirements.txt

```

### Scripts for training and testing
Before running the scripts, please modify them according to the specific model or language being used. <br>
To run the training script, use the following command:
```bash
./scripts/train-script.sh
```

Likewise, to run the script for testing (evaluation), use the following command:
```bash
./scripts/evaluate-script.sh
```

## Training and Evaluation
The following sections describe the key experiments and activities performed during the project:

### Initial Training and Evaluation
- Trained and evaluated Whisper models using PEFT and LoRA techniques
- Performed evaluations without prior training

### Grid Search
- Conducted grid search on learning rate and weight decay for Whisper model versions (whisper-large-v2, whisper-large-v3)
- The best models were selected using eval_loss metric on validation dataset
- Evaluated the best models from the grid search experiments

### Normalization and Spell Checking
- Performed normalization and spell checking for words in the training and test datasets
- Stored dataset words and their frequencies in a JSON file for spell checking analysis
- Evaluated models using normalization and spell checking

### Model Training
- Trained the best models to 16k steps to achieve lower error rate

<br>

## Results
### whisper-large-v2
All models used Batch Size: 16, Gradient Accumulation Steps: 1 and Max Steps: 4000.


| weight_decay | eval_loss           |
|--------------|---------------------|
| 0.001        | 0.12021409720182419 |
| 0.01         | 0.1278465837240219  |
| 0.1          | 0.12354397773742676 |
| 0.5          | 0.1667046844959259  |

The best value for `weight_decay` is **0.001**, and we will use it for selecting the best `learning_rate`.

| learning_rate | eval_loss           |
|---------------|---------------------|
| 5e-4          | 0.13173289597034454 |
| 7e-4          | 0.12321674823760986 |
| 9e-4          | 0.12281513214111328 |
| 2e-3          | 0.1383449286222458  |
| 5e-3          | 2.9641551971435547  |

The best model had `weight_decay = 1e-3` and `learning_rate = 9e-4`
After changing `gradient_accumulation_steps` to 2 and training the best model to 16000 steps, `eval_loss` fell to **0.11181466281414032**.

- **WER = 7.03**, **CER = 3.232** without spell checker
- **WER = 6.941**, **CER = 3.252** with spell checker


### whisper-large-v3
All models used Batch Size: 16, Gradient Accumulation Steps: 2 and Max Steps: 16000.

| learning_rate | weight_decay | eval_loss           | steps_trained |
|---------------|--------------|---------------------|---------------|
| 6e-4          | 1e-3         | 0.14830487966537476 | 2000          |
| 6e-4          | 5e-3         | 0.12888120114803314 | 5000          |
| 9e-4          | 1e-3         | 0.13642661273479462 | 5500          |
| 9e-4          | 5e-3         | 0.12117470800876617 | 9000          |
| 2e-3          | 1e-3         | 0.20787277817726135 | 2500          |
| 2e-3          | 5e-3         | 0.1882324069738388  | 4500          |

The best model had `learning_rate = 9e-4`, and `weight_decay = 5e-3`. 

- **WER = 7.581**, **CER = 3.466** without spell checker
- **WER = 7.387**, **CER = 3.347** with spell checker


<br>

## License
MIT License
