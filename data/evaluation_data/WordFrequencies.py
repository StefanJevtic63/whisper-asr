import json
import os
from datasets import load_from_disk

from data.evaluation_data.SerbianCyrillicNormalizer import SerbianCyrillicNormalizer

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DIR_PATH, "../datasets/train_validation_transcription_v3")
DATASET_FREQUENCIES_PATH = os.path.join(DIR_PATH, "dataset-word-frequencies.json")

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


if __name__ == "__main__":
    word_frequencies = WordFrequencies().calculate()

    # save the words and frequencies to the output file
    with open(DATASET_FREQUENCIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(word_frequencies, f, ensure_ascii=False, indent=4)