from srbai.SintaktickiOperatori.spellcheck import SpellCheck
import multiprocessing

class SerbianSpellChecker:
    def __init__(self, predictions):
        self.predictions = predictions
        self.sc = SpellCheck('sr-cyrillic')
        self.num_workers = 10

    def process_lines(self, lines):
        result = []
        for line in lines:
            words = []
            for word in line.split():
                correction = self.sc.spellcheck(word)
                words.append(correction if correction else word)
            result.append(' '.join(words))
        return result

    def spell_check(self):
        # equally divide amount of work
        workers_batch_size = len(self.predictions) // self.num_workers
        rest = len(self.predictions) % workers_batch_size

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