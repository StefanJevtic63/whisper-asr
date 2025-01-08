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
        for row in dataset:
            sentence = self.normalizer(row)
            for word in sentence.split():
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    def calculate(self):
        result = {}

        train_dataset = self.dataset["train"]
        self.add_words(result, train_dataset["transcription"])


        # validation dataset
        validation_dataset = self.dataset["validation"]
        self.add_words(result, validation_dataset["transcription"])

        return result
