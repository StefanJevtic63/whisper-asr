import numpy as np
import multiprocessing

class SpellChecker:
    """
    A class for performing spell checking on a set of predictions using a trie-based dictionary.

    :param list[str] predictions: List of predictions to be spell checked
    :param dict word_frequencies: Dictionary of word frequencies
    :param str dictionary_path: Path to the text file containing the dictionary of words
    :param int max_cost: Maximum allowed cost for edit distance (default is 2)
    """

    def __init__(self, predictions, word_frequencies, dictionary_path, max_cost=2):
        self.trie = {}
        self.max_cost = max_cost
        self.word_frequencies = word_frequencies
        self.load_dictionary(dictionary_path)
        self.predictions = predictions
        self.num_workers = 10

    def load_dictionary(self, dictionary):
        """
        Load words from a dictionary file into the trie structure.

        :param str dictionary: Path to the dictionary file containing words
        """

        with open(dictionary, 'r', encoding='utf-8') as f:
            for line in f:
                self.add_word(line.strip())

    def add_word(self, word):
        """
        Add a word to the trie structure.

        :param str word: Word to be added to the trie
        """

        node = self.trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = word

    def find(self, word):
        """
        Checks if a word exists in the trie structure.

        :param str word: Word to be checked
        :return: True if the word exists, False otherwise
        :rtype: bool
        """

        node = self.trie
        for char in word:
            if char not in node:
                return False
            node = node[char]

        return '$' in node

    def search_trie(self, node, word, original_word_len, cost, results):
        """
        Search for corrections in the trie that are within the maximum allowed cost.

        :param dict node: Current node in the trie
        :param str word: Word being searched
        :param int original_word_len: Length of the original word
        :param int cost: Current edit distance cost
        :param list[str] results: List to store found corrections
        """

        if cost > self.max_cost:
            return

        # length of the correction must be less than max cost
        if '$' in node and abs(len(node['$']) - original_word_len) <= self.max_cost:
            results.append(node['$'])
        for c in node:
            # end of the word
            if c == '$':
                continue

            # process the tails of strings
            if word and c == word[0]:
                self.search_trie(node[c], word[1:], original_word_len, cost, results)
            else:
                # substitute
                self.search_trie(node[c], word[1:], original_word_len, cost + 1, results)
                # insert
                self.search_trie(node[c], word, original_word_len, cost + 1, results)
                # delete
                if len(word) > 0:
                    self.search_trie(node[c], word[1:], original_word_len, cost + 1, results)

                # handle insertion to create longer words
                if cost + 1 <= self.max_cost:
                    self.search_trie(node, word[1:], original_word_len, cost + 1, results)
                    self.search_trie(node[c], word, original_word_len, cost + 1, results)

    def lev_distance(self, s, t):
        """
        Calculate the Levenshtein distance between two strings.

        :param str s: First string
        :param str t: Second string
        :return: The Levenshtein distance between the two strings
        :rtype: int
        """

        m, n = len(s), len(t)
        d = np.zeros((m+1, n+1), dtype=int)

        # source prefixes can be transformed into empty string by dropping all characters
        for i in range(m+1):
            d[i,0] = i

        # target prefixes can be reached from empty source prefix by inserting every character
        for j in range(n+1):
            d[0,j] = j

        for j in range(1, n+1):
            for i in range(1, m+1):
                if s[i-1] == t[j-1]:
                    substitution_cost = 0
                else:
                    substitution_cost = 1

                d[i,j] = min(d[i-1,j] + 1,                   # delete
                            d[i,j-1] + 1,                   # insert
                            d[i-1,j-1] + substitution_cost) # substitute

        return d[m,n]

    def find_closest(self, word, candidates):
        """
        Find the closest word from the list of candidates using Levenshtein distance.

        :param str word: The original word
        :param list[str] candidates: List of candidate words
        :return: The closest word from the candidates
        :rtype: str
        """

        min_distance = float('inf')
        closest_words = []
        for candidate in candidates:
            distance = self.lev_distance(word, candidate)
            if distance < min_distance:
                min_distance = distance
                closest_words = [candidate]
            elif distance == min_distance:
                closest_words.append(candidate)

        if not closest_words:
            return word

        closest_words_frequencies = {}
        for closest_word in closest_words:
            if closest_word in self.word_frequencies:
                closest_words_frequencies[closest_word] = self.word_frequencies[closest_word]
            else:
                closest_words_frequencies[closest_word] = 0

        return max(closest_words_frequencies, key=closest_words_frequencies.get)

    def spell_check_word(self, word):
        """
        Perform spell check for a given word.

        :param str word: The word to be checked
        :return: The corrected word
        :rtype: str
        """

        results = []
        self.search_trie(self.trie, word, len(word), 0, results)
        results = list(set(val for val in results))

        # try to split the word
        split_results = []
        for i in range(1, len(word)):
            part1, part2 = word[:i], word[i:]
            if self.find(part1) and self.find(part2):
                split_results.append(f"{part1} {part2}")
        results.extend(split_results)

        return self.find_closest(word, results)

    def combine_adjacent_words(self, word1, word2):
        """
        Combine two adjacent words if they form a valid word in the dictionary.

        :param str word1: The first word
        :param str word2: The second word
        :return: The combined word if valid, None otherwise
        :rtype: str or None
        """

        result = word1 + " " + word2
        if self.find(result):
            return result
        else:
            return None

    def process_lines(self, lines):
        """
        Process lines of text, performing spell checks on each word.

        :param list[str] lines: List of lines to be processed
        :return: List of processed lines with corrected words
        :rtype: list[str]
        """

        result = []
        for line in lines:
            words = []
            for word in line.split():
                correction = self.spell_check_word(word)
                words.append(correction)
            result.append(' '.join(words))
        return result

    def spell_check(self):
        """
        Perform spell check on all predictions using multiprocessing.

        :return: List of spell-checked predictions
        :rtype: list[str]
        """

        predictions_len = len(self.predictions)
        if self.num_workers > predictions_len:
            self.num_workers = predictions_len
            workers_batch_size = 1
            rest = 0
        else:
            workers_batch_size = predictions_len // self.num_workers
            rest = predictions_len % workers_batch_size

        partial_predictions = []
        begin = 0
        for i in range(self.num_workers):
            end = begin + workers_batch_size + (1 if i < rest else 0)
            partial_predictions.append(self.predictions[begin:end])
            begin = end

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            result = pool.map(self.process_lines, partial_predictions)

        # concat the predictions into one list
        final_result = []
        for lst in result:
            final_result.extend(lst)

        return final_result