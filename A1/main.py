from counter import NGramCounter
from perplexity import Perplexity
from preprocessor import Preprocessor
from language_model import LanguageModel
import numpy as np


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
    k_values = np.linspace(0.01, 10, num=100)
    discount_values = [0,0.15,0.25,0.5,0.65,0.75,1]
    backoff_values =  [0.1, 0.2, 0.4, 0.6, 0.7, 1.0]

    # for k in k_values:
    #     unigram_model = LanguageModel(unigram_counter, vocabulary_size, k=k, backoff=0.4)
    #     bigram_model = LanguageModel(bigram_counter, vocabulary_size, k=k, backoff=0.4)

    #     perplexity_unigram = Perplexity(unigram_model, preprocessor)
    #     perplexity_bigram = Perplexity(bigram_model, preprocessor)
    #     print("k = ", k)
    #     print(f"Unigram Un-Smoothed Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'ngram'):.2f}")
    #     print(f"Bigram Un-Smoothed Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'ngram'):.2f}")

    #     print(f"Unigram Add-k Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'add-k'):.2f}")
    #     print(f"Bigram Add-k Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'add-k'):.2f}")
    
    # for backoff in backoff_values:
    #     unigram_model = LanguageModel(unigram_counter, vocabulary_size, k=0.1, backoff=backoff)
    #     bigram_model = LanguageModel(bigram_counter, vocabulary_size, k=0.1, backoff=backoff)

    #     perplexity_unigram = Perplexity(unigram_model, preprocessor)
    #     perplexity_bigram = Perplexity(bigram_model, preprocessor)
    #     print("backoff = ", backoff)
    #     print(f"Unigram Backoff Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'backoff'):.2f}")
    #     print(f"Bigram Backoff Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'backoff'):.2f}")

    # for discount in discount_values:
    #     unigram_model = LanguageModel(unigram_counter, vocabulary_size, k=0.1, backoff=0.4, discount=discount)
    #     bigram_model = LanguageModel(bigram_counter, vocabulary_size, k=0.1, backoff=0.4, discount=discount)

    #     perplexity_unigram = Perplexity(unigram_model, preprocessor)
    #     perplexity_bigram = Perplexity(bigram_model, preprocessor)
    #     print("discount = ", discount)
    #     print(f"Unigram Absolute discounting Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'discounting'):.2f}")
    #     print(f"Bigram Absolute discounting Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'discounting'):.2f}")

    lambda_values = [
    [0.05, 0.95], [0.10, 0.90], [0.15, 0.85], [0.20, 0.80], [0.25, 0.75],
    [0.30, 0.70], [0.35, 0.65], [0.40, 0.60], [0.45, 0.55], [0.50, 0.50],
    [0.55, 0.45], [0.60, 0.40], [0.65, 0.35], [0.70, 0.30], [0.75, 0.25],
    [0.80, 0.20], [0.85, 0.15], [0.90, 0.10], [0.95, 0.05], [0.99, 0.01] ]

    for lambdas in lambda_values:
        lambda_1, lambda_2 = lambdas[0], lambdas[1]
        unigram_model = LanguageModel(unigram_counter, vocabulary_size, k=0.1, backoff=0.4, discount=0.15,lambda_1=lambda_1, lambda_2=lambda_2)
        bigram_model = LanguageModel(bigram_counter, vocabulary_size, k=0.1, backoff=0.4, discount=0.15,lambda_1=lambda_1, lambda_2=lambda_2)
        perplexity_unigram = Perplexity(unigram_model, preprocessor)
        perplexity_bigram = Perplexity(bigram_model, preprocessor)
        print(f"Unigram Interpolation Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'interpolation'):.2f}")
        print(f"Bigram Interpolation Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'interpolation'):.2f}")


    # unigram_model = LanguageModel(unigram_counter, vocabulary_size, k=0.1)
    # bigram_model = LanguageModel(bigram_counter, vocabulary_size, k=0.1)

    # Compute perplexity for each model
    # perplexity_unigram = Perplexity(unigram_model, preprocessor)
    # perplexity_bigram = Perplexity(bigram_model, preprocessor)

    # print(f"Unigram Un-Smoothed Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'ngram'):.2f}")
    # print(f"Bigram Un-Smoothed Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'ngram'):.2f}")

    # print(f"Unigram Add-k Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'add-k'):.2f}")
    # print(f"Bigram Add-k Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'add-k'):.2f}")

    print(f"Unigram Laplace Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'laplace'):.2f}")
    print(f"Bigram Laplace Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'laplace'):.2f}")


if __name__ == "__main__":
    main()
