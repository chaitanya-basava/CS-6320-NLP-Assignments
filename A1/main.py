from counter import NGramCounter
from perplexity import Perplexity
from preprocessor import Preprocessor
from language_model import LanguageModel


def main():
    # Load training data
    with open('./A1_DATASET/train.txt', 'r', encoding='utf-8') as file:
        train_data = file.readlines()

    # Load development data
    with open('./A1_DATASET/val.txt', 'r', encoding='utf-8') as file:
        dev_data = file.readlines()

    preprocessor = Preprocessor(2)
    preprocessor.build_vocab(train_data)

    # Preprocess training data
    train_tokens = [preprocessor.preprocess(sentence.strip()) for sentence in train_data]

    # counters
    unigram_counter = NGramCounter(n=1)
    bigram_counter = NGramCounter(n=2)

    # Train models
    for tokens in train_tokens:
        unigram_counter.count_ngrams(tokens)
        bigram_counter.count_ngrams(tokens)

    vocabulary_size = len(set(token for tokens in train_tokens for token in tokens))

    # Create language models
    unigram_model = LanguageModel(unigram_counter, vocabulary_size, k=0.1)
    bigram_model = LanguageModel(bigram_counter, vocabulary_size, k=0.1)

    # Compute perplexity for each model
    perplexity_unigram = Perplexity(unigram_model, preprocessor)
    perplexity_bigram = Perplexity(bigram_model, preprocessor)

    print(f"Unigram Un-Smoothed Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'ngram'):.2f}")
    print(f"Bigram Un-Smoothed Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'ngram'):.2f}")

    print(f"Unigram Add-k Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'add-k'):.2f}")
    print(f"Bigram Add-k Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'add-k'):.2f}")

    print(f"Unigram Laplace Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'laplace'):.2f}")
    print(f"Bigram Laplace Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'laplace'):.2f}")


if __name__ == "__main__":
    main()
