import argparse
import subprocess

import torch

from WhisperASR import WhisperASR


def env_check():
    """Checks if the GPU is available and prints the corresponding information."""

    # check if GPU is available
    print(torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA version:', torch.version.cuda)

    # print GPU info
    try:
        gpu_info = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            print('Not connected to a GPU')
        else:
            print('An error occurred: ', e)
    except FileNotFoundError:
        print('nvidia-smi not found. Is the Nvidia driver installed?')
    else:
        print(gpu_info)


if __name__ == "__main__":

    env_check()

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train WhisperASR model with a specified language and a Hugging Face dataset.')
    parser.add_argument('--language', type=str, required=True,
                        help="Language to train the Whisper ASR model with (e.g., 'Serbian')")
    parser.add_argument('--language_code', type=str, required=True,
                        help="Language code for the Whisper ASR model based on the data set. (e.g., 'sr' for Common Voice)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the trained model')

    parser.add_argument('--model_name', type=str, default="openai/whisper-small",
                        help="Model name from Hugging Face or custom path (default: 'openai/whisper-small')")
    parser.add_argument('--existing_model', action='store_true',
                        help="Flag to indicate an existing model is being used (default: False)")
    parser.add_argument('--save_to_hf', action='store_true',
                        help="Flag to indicate saving the model to Hugging Face Hub (default: False).")
    parser.add_argument('--dataset_name', type=str,  default="mozilla-foundation/common_voice_17_0",
                        help="Dataset name to train on (default: 'mozilla-foundation/common_voice_17_0')")
    parser.add_argument("--ref_key", type=str, default="sentence",
                        help="Key in dataset for reference data (default: 'sentence' - matches with Common Voice)")

    args = parser.parse_args()

    # train the model with command-line arguments
    model = WhisperASR(model_name=args.model_name,
                       language=args.language, language_code=args.language_code,
                       existing_model=args.existing_model, save_to_hf=args.save_to_hf,
                       output_dir=args.output_dir, dataset_name=args.dataset_name, ref_key=args.ref_key)
    model.train()
