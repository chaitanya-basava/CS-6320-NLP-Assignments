from collections import defaultdict
from typing import List, Dict, Tuple


class NGramCounter:
    def __init__(self, n: int):
        self.n = n
        self.total_tokens = 0
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

    def count_ngrams(self, tokens: List[str]) -> None:
        """
        Count n-grams in the given list of tokens.
        :param tokens: List of preprocessed tokens.
        :return: None
        """
        self.total_tokens += len(tokens[1:-1])
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i: i+self.n])
            self.ngram_counts[ngram] += 1

            if self.n > 1:
                context = ngram[:-1]
                self.context_counts[context] += 1
