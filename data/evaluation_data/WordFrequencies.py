import sys, os
from datasets import load_from_disk

from data.evaluation_data.SerbianCyrillicNormalizer import SerbianCyrillicNormalizer

CURRENT_DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIRECTORY_PATH, "../datasets/train_validation_dataset_transcription_v2")


class WordFrequencies:
    """
    A class to compute the frequency of word occurrences within a dataset.

    This class loads a dataset, normalizes the text, and computes word frequency distributions for given datasets.
    """

    def __init__(self):
        self.dataset = load_from_disk(DATASET_PATH)
        self.normalizer = SerbianCyrillicNormalizer()

    def add_words(self, dictionary, dataset):
        """
        Computes the frequency of word occurrences within the specified dataset.

        :param dictionary: The dictionary representing word occurrences
        :type dictionary: dict
        :param dataset: The given dataset
        :type dataset: Dataset

        :return: Updated dictionary with the count of word occurrences
        :rtype: dict
        """

        for row in dataset:
            sentence = self.normalizer(row)
            for word in sentence.split():
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1
        return dictionary

    def calculate(self):
        """
        Computes the frequency of word occurrences within the train and validation datasets.

        :return:Dictionary with the count of word occurrences from both datasets
        :rtype: dict
        """

        result = {}

        train_dataset = self.dataset["train"]
        self.add_words(result, train_dataset["transcription"])

        validation_dataset = self.dataset["validation"]
        self.add_words(result, validation_dataset["transcription"])

        return result
