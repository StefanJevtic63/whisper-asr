from datasets import Audio, load_dataset, concatenate_datasets, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from WhisperASR import HF_API_KEY

class Datasets:
    """Class for merging multiple datasets."""

    def __init__(self, feature_extractor, tokenizer, hf_api_key, ref_key):
        """Initiate the dataset merging class.

        Args:
            feature_extractor (FeatureExtractor): Whisper feature extractor
            tokenizer (Tokenizer): Whisper tokenizer
            hf_api_key (str): The HuggingFace API key
            ref_key (str): The key to the reference data in the dataset
        """

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.hf_api_key = hf_api_key
        self.ref_key = ref_key

        # load datasets
        self.parlaspeech = self._parlaspeech()
        self.fleurs = self._fleurs()
        self.common_voice = self._common_voice()

    def _prepare_data(self, batch):
        """Converts audio files to the model's input feature format and encodes the target texts.

        Args:
            batch (dict): A batch of audio and text data.
        """

        # load and resample audio data from 48kHz to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch[self.ref_key]).input_ids

        return batch

    def _generalize_parlaspeech(self, parlaspeech):
        """Generalizes ParlaSpeech-SR dataset.

        Args:
            parlaspeech (DatasetDict): The dataset to be generalized

        Returns:
            DatasetDict
        """


        # only consider input audio and transcribed text to generalize dataset as much as possible
        parlaspeech = parlaspeech.remove_columns(
            ['id', 'text', 'text_cyrillic', 'text_normalised', 'words',
             'audio_length', 'date', 'speaker_name', 'speaker_gender', 'speaker_birth',
             'speaker_party', 'party_orientation', 'party_status']
        )

        # generalize the transcription column name
        parlaspeech = parlaspeech.rename_column("text_cyrillic_normalised", "transcription")
        return parlaspeech

    def _generalize_fleurs(self, fleurs):
        """Generalizes Google/Fleurs dataset.

        Args:
            fleurs (DatasetDict): The dataset to be generalized

        Returns:
            DatasetDict
        """

        # only consider input audio and transcribed text to generalize dataset as much as possible
        return fleurs.remove_columns(
            ['id', 'num_samples', 'path', 'raw_transcription',
             'gender', 'lang_id', 'language', 'lang_group_id']
        )

    def _generalize_common_voice(self, common_voice):
        """Generalizes Common Voice dataset.

        Args:
            common_voice (DatasetDict): The dataset to be generalized

        Returns:
            DatasetDict
        """

        # only consider input audio and transcribed text to generalize dataset as much as possible
        common_voice = common_voice.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender",
             "locale", "path", "segment", "up_votes", "variant"]
        )

        # generalize the transcription column name
        common_voice = common_voice.rename_column("sentence", "transcription")
        return common_voice

    def _generalize(self, dataset, dataset_name):
        """Generalizes a custom dataset.

        Args:
            dataset (DatasetDict): The dataset to be generalized
            dataset_name (str): The name of the custom dataset

        Returns:
            DatasetDict
        """

        if dataset_name == "classla/ParlaSpeech-RS":
            return self._generalize_parlaspeech(dataset)
        elif dataset_name == "google/fleurs":
            return self._generalize_fleurs(dataset)
        else:
            return self._generalize_common_voice(dataset)

    def _downsample(self, dataset, split):
        """Downsamples audio data to 16kHz.

        Args:
            dataset (DatasetDict): The dataset to be downsampled
            split (str): The split of a dataset

        Returns:
            DatasetDict
        """

        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.map(self._prepare_data, remove_columns=dataset.column_names[split], num_proc=4)
        return dataset

    def _load_dataset(self, path, name = None, split = None):
        """Load a custom dataset.

        Args:
            path (str): The name of the dataset
            name (str): The language code
            split (str): The dataset split

        Returns:
            Dataset if split is not None or DatasetDict with each split if split is None
        """

        return load_dataset(path, name, token=self.hf_api_key, split=split, trust_remote_code=True)

    def _fill_dataset(self, split, train_dataset = None, validation_dataset = None, test_dataset = None,
                      path = None, name = None):
        """Initializes a dataset by a split.

        Args:
            split (str): The dataset split
            train_dataset (Dataset, DatasetDict): The train dataset
            validation_dataset (Dataset, DatasetDict): The validation dataset
            test_dataset (Dataset, DatasetDict): The test dataset
            path (str): The name of the dataset
            name (str): The language code

        Returns:
              DatasetDict
        """


        dataset = DatasetDict()
        if split == "train":
            dataset["train"] = train_dataset if train_dataset else (
                    self._load_dataset(path, name, "train"))
            dataset["validation"] = validation_dataset if validation_dataset else (
                    self._load_dataset(path, name, "validation"))
        else:
            dataset["test"] = test_dataset if test_dataset else (
                    self._load_dataset(path, name, "test"))

        return dataset

    def _load_and_generalize_split(self, split, train_dataset = None, validation_dataset = None, test_dataset = None,
                                     path = None, name = None):
        """Loads and generalizes custom dataset by a split.

        Args:
            split (str): The dataset split
            train_dataset (Dataset, DatasetDict): The train dataset
            validation_dataset (Dataset, DatasetDict): The validation dataset
            test_dataset (Dataset, DatasetDict): The test dataset
            path (str): The name of the dataset
            name (str): The language code

        Returns:
            DatasetDict
        """

        dataset = self._fill_dataset(
            split=split,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            path = path,
            name = name
        )

        dataset = self._generalize(dataset, path)
        dataset = self._downsample(dataset, split)
        return dataset

    def _load_and_generalize_dataset(self, train_dataset = None, validation_dataset = None, test_dataset = None,
                                     path = None, name = None):
        """Loads and generalizes custom dataset.

        Args:
            train_dataset (Dataset, DatasetDict): The train dataset
            validation_dataset (Dataset, DatasetDict): The validation dataset
            test_dataset (Dataset, DatasetDict): The test dataset
            path (str): The name of the dataset
            name (str): The language code

        Returns:
            DatasetDict
        """

        # load and generalize the dataset for train split
        dataset_train_val = self._load_and_generalize_split(
            split="train",
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            path=path,
            name=name
        )

        # load and generalize the dataset for test split
        dataset_test = self._load_and_generalize_split(
            split="test",
            test_dataset=test_dataset,
            path=path,
            name=name
        )

        return DatasetDict({
            "train": dataset_train_val["train"],
            "validation": dataset_train_val["validation"],
            "test": dataset_test["test"]
        })

    def _parlaspeech(self):
        """Prepare ParlaSpeech-RS dataset."""

        # features: ['id', 'audio', 'text', 'text_cyrillic', 'text_normalised', 'text_cyrillic_normalised',
        #            'words', 'audio_length', 'date', 'speaker_name', 'speaker_gender', 'speaker_birth',
        #            'speaker_party', 'party_orientation', 'party_status']

        # load the dataset
        path = "classla/ParlaSpeech-RS"
        dataset_parlaspeech = self._load_dataset(path=path)

        # split the dataset on training (80%) and temporary set (20%)
        train_temp_split = dataset_parlaspeech["train"].train_test_split(test_size=0.2)
        train_dataset = train_temp_split["train"]
        temp_dataset = train_temp_split["test"]

        # split temporary dataset on validation (50% of 20%) and test (50% of 20%) dataset
        validation_test_split = temp_dataset.train_test_split(test_size=0.5)
        validation_dataset = validation_test_split["train"]
        test_dataset = validation_test_split["test"]

        return self._load_and_generalize_dataset(train_dataset, validation_dataset, test_dataset, path)

    def _fleurs(self):
        """Prepare Fleurs dataset.

        Returns:
            DatasetDict
        """

        # features: ['id', 'num_samples', 'path', 'audio', 'transcription', 'raw_transcription', 'gender',
        #            'lang_id', 'language', 'lang_group_id']

        return self._load_and_generalize_dataset(path="google/fleurs", name="sr_rs")

    def _common_voice(self):
        """Prepare Common Voice dataset for training and validation.

        Returns:
            DatasetDict
        """

        return self._load_and_generalize_dataset(path="mozilla-foundation/common_voice_17_0", name="sr")

    def _combine_datasets_by_split(self, split):
        """Combines datasets for a certain split.

        Args:
            split (str): The dataset split

        Returns:
            [Dataset]
        """

        combined_dataset = concatenate_datasets([
            self.parlaspeech[split],
            self.fleurs[split],
            self.common_voice[split]
        ])

        return combined_dataset

    def combine_datasets_train_val(self):
        """Creates a combined dataset with train and validation splits.

        Returns:
            DatasetDict
        """

        data = DatasetDict()
        data["train"] = self._combine_datasets_by_split("train")
        data["validation"] = self._combine_datasets_by_split("validation")

        return data

    def combine_datasets_test(self):
        """Creates a combined dataset with test split.

        Returns:
            DatasetDict
        """

        data = DatasetDict()
        data["test"] = self._combine_datasets_by_split("test")

        return data

if __name__ == "__main__":
    # initiate parameters
    model_name = "openai/whisper-large-v2"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, language="Serbian", task="transcribe")

    # load datasets
    datasets = Datasets(feature_extractor, tokenizer, HF_API_KEY, "transcription")
    train_val_dataset = datasets.combine_datasets_train_val()
    test_dataset = datasets.combine_datasets_test()

    # save datasets for future use
    path_to_train_val_dataset = "train_validation_dataset"
    path_to_test_dataset = "test_dataset"

    train_val_dataset.save_to_disk(path_to_train_val_dataset)
    test_dataset.save_to_disk(path_to_test_dataset)