import re
from typing import List
from collections import defaultdict


class Preprocessor:
    def __init__(self, min_count: int = 2):
        self.start_token = "<s>"
        self.end_token = "</s>"

        self.unk_token = "<UNK>"
        self.min_count = min_count

        self.word_counts = defaultdict(int)

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess the input text.
        :param text: Input text to preprocess.
        :return: List of preprocessed tokens.
        """
        # Convert to lowercase
        text = text.lower()

        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Split into tokens
        tokens = text.split()

        # Add start and end tokens
        tokens = [self.start_token] + tokens + [self.end_token]

        # Replace tokens with low frequency with <UNK>
        tokens = [self.unk_token if self.word_counts[token] < self.min_count else token for token in tokens]

        return tokens

    def build_vocab(self, texts: List[str]):
        """
        Build a vocabulary from a list of texts.
        :param texts: List of input texts.
        :return: None
        """
        self.word_counts = defaultdict(int)
        for text in texts:
            tokens = text.split()
            for token in tokens:
                self.word_counts[token] += 1
