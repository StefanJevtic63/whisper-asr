import re
from num2words import num2words
from transliterate import translit

class SerbianCyrillicNormalizer:
    """Class for normalizing the text in Serbian language written in Cyrillic alphabet."""

    def __init__(self):
        """Initate the normalization class."""

    def __call__(self, text):
        # remove punctuation from the entire text
        text = re.sub(r'[^\w\s]', ' ', text)

        # convert the entire text to lowercase
        text = text.lower()

        # split the text into words
        words = text.split()

        # normalize each word
        normalized_words = []
        for word in words:
            # if the word is a number, convert it to words instead
            if word.isdigit():
                word = num2words(int(word), lang='sr')
                word_parts = word.split()
                if len(word_parts) > 1:
                    word = ' '.join(word_parts[:-1] + ['i'] + word_parts[-1:])

            normalized_words.append(word)

        # join the normalized words back into a sentence
        normalized_text = ' '.join(normalized_words)

        # convert characters to Serbian Cyrillic alphabet
        normalized_text = translit(normalized_text, 'sr')

        # trim the output
        normalized_text = normalized_text.strip()

        return normalized_text