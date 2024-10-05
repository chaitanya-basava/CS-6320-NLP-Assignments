import math
from typing import List

from preprocessor import Preprocessor
from language_model import LanguageModel


class Perplexity:
    def __init__(self, language_model: LanguageModel, preprocessor: Preprocessor):
        self.language_model = language_model
        self.preprocessor = preprocessor

    def compute_perplexity(self, dev_set: List[str], _type: str) -> float:
        """
        Compute the perplexity of the language model on the development set.
        :param dev_set: List of sentences in the development set.
        :param _type: The type of smoothing to use.
        :return: Perplexity of the model on the development set.
        """
        total_log_probability = 0
        total_words = 0

        for sentence in dev_set:
            tokens = self.preprocessor.preprocess(sentence)
            sentence_log_prob = self.sentence_log_probability(tokens, _type)
            total_log_probability += sentence_log_prob
            total_words += len(tokens) - 2

        average_log_probability = total_log_probability / total_words
        perplexity = math.pow(2, -average_log_probability)
        return perplexity

    def sentence_log_probability(self, tokens: List[str], _type: str) -> float:
        """
        Compute the log probability of a sentence.
        :param tokens: List of tokens in the sentence.
        :param _type: The type of smoothing to use.
        :return: Log probability of the sentence.
        """
        log_prob = 0
        n = self.language_model.ng_counter.n

        for i in range(n-1, len(tokens)):
            context = tuple(tokens[max(0, i - n + 1): i])
            prob = self.language_model.probability(_type, *context, tokens[i])
            log_prob += math.log(prob) if prob > 0 else float('-inf')

        return log_prob
