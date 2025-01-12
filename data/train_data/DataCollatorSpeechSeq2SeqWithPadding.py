from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    A data collator for speech sequence-to-sequence models with padding.

    :param Any processor: The processor used for feature extraction and tokenization
    """

    processor: Any

    def __call__(self, features):
        """
        Prepare a batch of input features and labels for speech sequence-to-sequence models.

        The method performs the following steps:
            - Splits inputs and labels, treating the audio inputs by returning torch tensors
            - Pads the input features to the maximum length
            - Pads the label sequences to the maximum length and replaces padding with -100 to correctly ignore the loss
            - Removes the BOS token if appended in the previous tokenization step
            - Moves all tensors to the GPU if available

        :param features: List of dictionaries containing input features and labels
        :type features: List[Dict[str, Union[List[int], torch.Tensor]]]

        :return: Batch of input features and labels as torch tensors
        :rtype: Dict[str, torch.Tensor]
        """

        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # move all tensors to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = {k: v.to(device) for k, v in batch.items()}

        return batch
