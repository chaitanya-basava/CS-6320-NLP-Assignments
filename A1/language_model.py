from typing import Tuple

from counter import NGramCounter
from preprocessor import Preprocessor


class LanguageModel:
    def __init__(self, ng_counter: NGramCounter, vocab_size: int, k: float = 1.0, backoff: float = 0.4, discount: float = 0.75,
                 lambda_1: float = 0.5, lambda_2: float = 0.5):
        self.k = k
        self.backoff = backoff
        self.discount = discount
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
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
        elif _type == 'backoff':
            return self.backoff_probability(*_tokens)
        elif _type == 'discounting':
            return self.absolute_discounting_probability(*_tokens)
        elif _type == 'interpolation':
            return self.interpolation_probability(*_tokens)  
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
    
    def unigram_probability(self, token: str) -> float:
        """
        Compute the probability of a unigram (1-token n-gram).
        This is separate from n-gram validation, which expects multiple tokens.
        """
        unigram_count = self.ng_counter.ngram_counts.get((token,), 0)  # Unigram count
        return unigram_count / self.ng_counter.total_tokens if self.ng_counter.total_tokens > 0 else 0



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
    
    def backoff_probability(self, *_tokens: str) -> float:
        """
        Compute the probability of an n-gram with backoff smoothing.
        :param _tokens: The tokens of the n-gram to compute the probability for.
        :return: The probability of the given n-gram with backoff smoothing.
        """
        ngram = tuple(_tokens)

        # Handle unigram case: If it's a unigram, apply Laplace/Add-k smoothing
        if len(ngram) == 1:
            token = ngram[0]
            unigram_count = self.ng_counter.ngram_counts.get((token,), 0)
            # Apply Laplace or Add-k smoothing (set k=1 for Laplace smoothing)
            smoothed_prob = (unigram_count + 1) / (self.ng_counter.total_tokens + self.vocab_size)
            return smoothed_prob

        # For higher-order n-grams (e.g., bigrams or trigrams), apply backoff
        context = ngram[:-1]
        ngram_count = self.ng_counter.ngram_counts.get(ngram, 0)
        context_count = self.ng_counter.context_counts.get(context, 0)

        if context_count > 0 and ngram_count > 0:
            # Return the higher-order n-gram probability
            return ngram_count / context_count
        else:
            # Backoff to lower-order n-gram (remove the first token)
            return self.backoff * self.backoff_probability(*ngram[1:])

    
    def absolute_discounting_probability(self, *_tokens: str) -> float:
        """
        Compute the probability of an n-gram using absolute discounting.
        :param _tokens: The tokens of the n-gram to compute the probability for.
        :return: The probability of the n-gram with absolute discounting.
        """
        ngram = self.__validate_ngram(*_tokens)
        
        # Context for the n-gram (all tokens except the last)
        context = ngram[:-1]
        context_count = self.ng_counter.context_counts.get(context, 0)

        if context_count == 0:
            # If context doesn't exist, back off to lower order n-gram (e.g., unigram probability)
            return self.backoff_probability(*_tokens)

        # Count of the n-gram
        ngram_count = self.ng_counter.ngram_counts.get(ngram, 0)

        # Apply discount to the n-gram count
        discounted_count = max(ngram_count - self.discount, 0)

        # Calculate the first part of the absolute discounting formula
        discounted_prob = discounted_count / context_count

        # Compute the number of distinct followers (tokens that follow the given context)
        num_context_followers = len({
            ngram[-1] for ngram in self.ng_counter.ngram_counts if ngram[:-1] == context
        })

        # Calculate lambda (remaining probability mass) for backoff
        lambda_weight = (self.discount * num_context_followers) / context_count if context_count > 0 else 0

        # Combine discounted probability and backed off lower-order probability
        lower_order_prob = self.backoff_probability(*_tokens[1:])
        return discounted_prob + lambda_weight * lower_order_prob
    
    def interpolation_probability(self, *_tokens: str) -> float:
        """
        Compute the probability of an n-gram using interpolation smoothing.
        :param _tokens: The tokens of the n-gram to compute the probability for.
        :return: The interpolated probability of the n-gram.
        """
        ngram = tuple(_tokens)

        # If it's a bigram model, calculate interpolated probability between bigram and unigram
        if len(ngram) == 2:
            bigram_prob = self.ngram_probability(*ngram)  # Get bigram probability
            unigram_prob = self.unigram_probability(ngram[-1])  # Get unigram probability for the last word

            # Interpolated probability: lambda_2 * P_bigram + lambda_1 * P_unigram
            return self.lambda_2 * bigram_prob + self.lambda_1 * unigram_prob

        # If it's a unigram model, just return the unigram probability
        elif len(ngram) == 1:
            return self.unigram_probability(*ngram)
        
        # For other n-grams (e.g., trigram), you could extend this logic if needed
        else:
            return 0  # Default return value for unsupported n-grams



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
