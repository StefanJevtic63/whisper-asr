import sys, os
from datasets import load_from_disk

from data.evaluation_data.SerbianCyrillicNormalizer import SerbianCyrillicNormalizer

CURRENT_DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIRECTORY_PATH, "../datasets/train_validation_dataset_transcription_v2")

class WordFrequencies:
    def __init__(self):
        self.dataset = load_from_disk(DATASET_PATH)
        self.normalizer = SerbianCyrillicNormalizer()

    def add_words(self, dictionary, dataset):
        """Calculates the number of word occurrences in the given dataset.

        Args:
            dictionary (dict): The dictionary representing word occurrences
            dataset (Dataset, DatasetDict): The given dataset

        Returns
            dict
        """

        for row in dataset:
            sentence = self.normalizer(row)
            for word in sentence.split():
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    def calculate(self):
        """Calculates the number of word occurrences in train and validation datasets.

        Returns:
            dict
        """

        result = {}

        train_dataset = self.dataset["train"]
        self.add_words(result, train_dataset["transcription"])

        # validation dataset
        validation_dataset = self.dataset["validation"]
        self.add_words(result, validation_dataset["transcription"])

        return result
