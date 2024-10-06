import re
from typing import List
import matplotlib.pyplot as plt
from collections import defaultdict


class Preprocessor:
    def __init__(self, min_count: int = 2):
        self.start_token = "<s>"
        self.end_token = "</s>"

        self.unk_token = "<UNK>"
        self.min_count = min_count

        self.word_counts = defaultdict(int)

    def preprocess(self, _text: str, unk: bool = True) -> List[str]:
        """
        Preprocess the input text.
        :param _text: Input text to preprocess.
        :param unk: treat unk words or not
        :return: List of preprocessed tokens.
        """
        # Convert to lowercase
        _text = _text.lower()

        # Replace multiple spaces with a single space
        _text = re.sub(r'\s+', ' ', _text)

        # Remove all punctuations
        _text = re.sub(r'[^\w\s]', '', _text)

        # Split into tokens
        tokens = _text.split()

        # Add start and end tokens
        tokens = [self.start_token] + tokens + [self.end_token]

        # Replace tokens with low frequency with <UNK>
        if unk:
            tokens = [self.unk_token if token not in self.word_counts else token for token in tokens]

        return tokens

    def build_vocab(self, texts: List[str]):
        """
        Build a vocabulary from a list of texts.
        :param texts: List of input texts.
        :return: None
        """
        self.word_counts = defaultdict(int)
        for _text in texts:
            for token in self.preprocess(_text, False):
                self.word_counts[token] += 1

        self.word_counts[self.unk_token] = 0
        for token in list(self.word_counts.keys()):
            if (token not in [self.start_token, self.end_token, self.unk_token]
                    and self.word_counts[token] < self.min_count):
                self.word_counts[self.unk_token] += self.word_counts.pop(token)

    def vocab_size(self):
        return len(self.word_counts)


def plot_histogram(counts, name):
    counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    # print(counts)

    words, counts = zip(*counts[:20])
    words = [' '.join(pair) for pair in words]

    plt.figure(figsize=(12, 8))
    plt.bar(words, counts, color='blue')
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.title('Histogram of Word Counts')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"./imgs/{name}")


if __name__ == "__main__":
    preprocessor = Preprocessor(min_count=0)

    # Load training data
    with open('./A1_DATASET/train.txt', 'r', encoding='utf-8') as file:
        train_data = file.readlines()

    preprocessor.build_vocab(train_data)

    # print(preprocessor.vocab_size())

    plot_histogram(preprocessor.word_counts, "unigram.png")
