from typing import Tuple

from counter import NGramCounter
from preprocessor import Preprocessor


class LanguageModel:
    def __init__(self, ng_counter: NGramCounter, vocab_size: int, k: float = 1.0):
        self.k = k
        self.ng_counter = ng_counter
        self.vocab_size = vocab_size

    def __validate_ngram(self, *_tokens: str) -> Tuple[str, ...]:
        """
        Validate the n-gram.
        :param _tokens: The tokens of the n-gram to validate.
        :return: None
        """
        ngram = tuple(_tokens)
        if len(ngram) != self.ng_counter.n:
            raise ValueError(f"Expected {self.ng_counter.n} tokens for n-gram probability calculation.")

        return ngram

    def probability(self, _type: str, *_tokens: str) -> float:
        """
        Compute the probability of an n-gram.
        :param _tokens: The tokens of the n-gram to compute the probability for.
        :param _type: The type of smoothing to use.
        :return: The probability of the given n-gram.
        """
        if _type == 'ngram':
            return self.ngram_probability(*_tokens)
        elif _type == 'laplace':
            return self.laplace_probability(*_tokens)
        elif _type == 'add-k':
            return self.add_k_probability(*_tokens)
        else:
            raise ValueError(f"Invalid type of smoothing: {_type}")

    def ngram_probability(self, *_tokens: str) -> float:
        """
        Compute the probability of an n-gram.
        :param _tokens: The tokens of the n-gram to compute the probability for.
        :return: The probability of the given n-gram.
        """
        ngram = self.__validate_ngram(*_tokens)

        denom = self.ng_counter.total_tokens if self.ng_counter.n == 1 else self.ng_counter.context_counts[ngram[:-1]]
        return self.ng_counter.ngram_counts[ngram] / denom if denom > 0 else 0

    def laplace_probability(self, *_tokens: str) -> float:
        """
        Compute the Laplace-smoothed probability of an n-gram.
        :param _tokens: The tokens of the n-gram to compute the probability for.
        :return: The Laplace-smoothed probability of the given n-gram.
        """
        ngram = self.__validate_ngram(*_tokens)

        denom = self.ng_counter.total_tokens if self.ng_counter.n == 1 else self.ng_counter.context_counts[ngram[:-1]]
        return (self.ng_counter.ngram_counts[ngram] + 1) / (denom + self.vocab_size)

    def add_k_probability(self, *_tokens: str) -> float:
        """
        Compute the add-k smoothed probability of an n-gram.
        :param _tokens: The tokens of the n-gram to compute the probability for.
        :return: The add-k smoothed probability of the given n-gram.
        """
        ngram = self.__validate_ngram(*_tokens)

        denom = self.ng_counter.total_tokens if self.ng_counter.n == 1 else self.ng_counter.context_counts[ngram[:-1]]
        return (self.ng_counter.ngram_counts[ngram] + self.k) / (denom + self.k * self.vocab_size)


if __name__ == "__main__":
    preprocessor = Preprocessor(min_count=0)  # no unknown token treatment
    text = "the students like the assignment."
    tokens = preprocessor.preprocess(text)

    unigram_counter = NGramCounter(n=1)
    bigram_counter = NGramCounter(n=2)

    unigram_counter.count_ngrams(tokens)
    bigram_counter.count_ngrams(tokens)

    vocabulary_size = len(preprocessor.word_counts)

    unigram_model = LanguageModel(unigram_counter, vocabulary_size, k=0.1)
    bigram_model = LanguageModel(bigram_counter, vocabulary_size, k=0.1)

    print("Un-smoothed probabilities:")

    print(f"p('the'): {unigram_model.probability('ngram', 'the'):.6f}")
    print(f"p('like'): {unigram_model.probability('ngram', 'like'):.6f}")

    print(f"p('like the'): {bigram_model.probability('ngram', 'like', 'the'):.6f}")
    print(f"p('the students'): {bigram_model.probability('ngram', 'the', 'students'):.6f}")

    print("Laplace-smoothed probabilities:")

    print(f"p('the'): {unigram_model.probability('laplace', 'the'):.6f}")
    print(f"p('like'): {unigram_model.probability('laplace', 'like'):.6f}")

    print(f"p('like the'): {bigram_model.probability('laplace', 'like', 'the'):.6f}")
    print(f"p('the students'): {bigram_model.probability('laplace', 'the', 'students'):.6f}")

    print("Add-k smoothed probabilities:")

    print(f"p('the'): {unigram_model.probability('add-k', 'the'):.6f}")
    print(f"p('like'): {unigram_model.probability('add-k', 'like'):.6f}")

    print(f"p('like the'): {bigram_model.probability('add-k', 'like', 'the'):.6f}")
    print(f"p('the students'): {bigram_model.probability('add-k', 'the', 'students'):.6f}")
