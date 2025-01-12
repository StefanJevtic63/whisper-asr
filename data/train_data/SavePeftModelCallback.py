import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class SavePeftModelCallback(TrainerCallback):
    """
    A custom callback for saving PEFT models during training with Hugging Face's Trainer.

    This callback ensures that the adapter model is saved separately at each checkpoint, and the default PyTorch model file is removed.

    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Save the PEFT model and clean up the default PyTorch model file during training.

        :param TrainingArguments args: Training arguments containing output directory information
        :param TrainerState state: Trainer state containing information like the global step
        :param TrainerControl control: Trainer control object to manage training control flow
        :param kwargs: Additional keyword arguments, including the model to be saved

        :return: Training control object to manage further training control flow
        :rtype: TrainerControl
        """

        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control