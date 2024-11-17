import re
from num2words import num2words
from transliterate import translit

class SerbianCyrillicNormalizer:
    """Class for normalizing the text in Serbian language written in Cyrillic alphabet."""

    def __init__(self):
        """Initate the normalization class."""

    def __call__(self, word):
        # remove punctuation
        word = re.sub(r'[^\w\s]', '', word)

        # convert characters to lowercase
        word = word.lower()

        # if the word is a number, convert it to words instead
        if word.isdigit():
            word = num2words(int(word), lang='sr')
            word_parts = word.split()
            if len(word_parts) > 1:
                word = ' '.join(word_parts[:-1] + ['i'] + word_parts[-1:])

        # convert characters to Serbian Cyrillic alphabet
        word = translit(word, 'sr')

        # trim the output
        word = word.strip()

        return word