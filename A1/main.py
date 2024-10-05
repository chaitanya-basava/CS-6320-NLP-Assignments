import numpy as np

from counter import NGramCounter
from perplexity import Perplexity
from language_model import LanguageModel
from preprocessor import Preprocessor, plot_histogram


# Load training data
with open('./A1_DATASET/train.txt', 'r', encoding='utf-8') as file:
    train_data = file.readlines()

# Load development data
with open('./A1_DATASET/val.txt', 'r', encoding='utf-8') as file:
    dev_data = file.readlines()


def main():
    preprocessor = Preprocessor(5)
    preprocessor.build_vocab(train_data)

    # Preprocess training data
    train_tokens = [preprocessor.preprocess(sentence.strip()) for sentence in train_data]

    vocabulary_size = preprocessor.vocab_size()
    print(f"Vocabulary size: {vocabulary_size}")

    # counters
    unigram_counter = NGramCounter(n=1)
    bigram_counter = NGramCounter(n=2)

    # Train models
    for tokens in train_tokens:
        unigram_counter.count_ngrams(tokens)
        bigram_counter.count_ngrams(tokens)

    plot_histogram(unigram_counter.ngram_counts, "unigram.png")
    plot_histogram(bigram_counter.ngram_counts, "bigram.png")

    # Create language models
    k_values = np.linspace(0.01, 0.9, num=10)

    unigram_model = LanguageModel(unigram_counter, vocabulary_size)
    bigram_model = LanguageModel(bigram_counter, vocabulary_size)

    perplexity_unigram = Perplexity(unigram_model, preprocessor)
    perplexity_bigram = Perplexity(bigram_model, preprocessor)

    for k in k_values:
        unigram_model.set_k(k)
        bigram_model.set_k(k)

        print("\nk = ", k)
        print(f"Unigram Add-k Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'add-k'):.2f}")
        print(f"Bigram Add-k Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'add-k'):.2f}")

    unigram_model = LanguageModel(unigram_counter, vocabulary_size)
    bigram_model = LanguageModel(bigram_counter, vocabulary_size)
    perplexity_unigram = Perplexity(unigram_model, preprocessor)
    perplexity_bigram = Perplexity(bigram_model, preprocessor)

    print("\nLaplace Smoothing:")
    print(f"Unigram Laplace Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'laplace'):.2f}")
    print(f"Bigram Laplace Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'laplace'):.2f}")

    print("\nUn-smoothed:")
    print(f"Un-smoothed Unigram Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'ngram'):.2f}")
    print(f"Un-smoothed Bigram Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'ngram'):.2f}")


if __name__ == "__main__":
    main()
