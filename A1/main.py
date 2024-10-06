import numpy as np

from counter import NGramCounter
from perplexity import Perplexity
from language_model import LanguageModel
from preprocessor import Preprocessor, plot_histogram
import matplotlib.pyplot as plt


# Load training data
with open('./A1_DATASET/train.txt', 'r', encoding='utf-8') as file:
    train_data = file.readlines()

# Load development data
with open('./A1_DATASET/val.txt', 'r', encoding='utf-8') as file:
    dev_data = file.readlines()


def main():
    missing_token_count = list(range(2, 11, 1))
    vocab_collect = []
    for token_count in missing_token_count:
        print("Missing Token Count: ",token_count)
        preprocessor = Preprocessor(token_count)
        preprocessor.build_vocab(train_data)

        # Preprocess training data
        train_tokens = [preprocessor.preprocess(sentence.strip()) for sentence in train_data]

        vocabulary_size = preprocessor.vocab_size()
        vocab_collect.append(vocabulary_size)
        print(f"Vocabulary size: {vocabulary_size}")


        # counters
        unigram_counter = NGramCounter(n=1)
        bigram_counter = NGramCounter(n=2)

        # Train models
        for tokens in train_tokens:
            unigram_counter.count_ngrams(tokens)
            bigram_counter.count_ngrams(tokens)

        plot_histogram(unigram_counter.ngram_counts, "unigram_"+str(token_count)+".png")
        plot_histogram(bigram_counter.ngram_counts, "bigram_"+str(token_count)+".png")

        # Create language models
        k_values = np.linspace(0.01, 0.9, num=10)

        unigram_model = LanguageModel(unigram_counter, vocabulary_size)
        bigram_model = LanguageModel(bigram_counter, vocabulary_size)

        perplexity_unigram = Perplexity(unigram_model, preprocessor)
        perplexity_bigram = Perplexity(bigram_model, preprocessor)

        unigram_perplexities_train = []
        bigram_perplexities_train = []
        unigram_perplexities_dev = []
        bigram_perplexities_dev = []

        for k in k_values:
            unigram_model.set_k(k)
            bigram_model.set_k(k)
            unigram_perplexities_train.append(perplexity_unigram.compute_perplexity(train_data, 'add-k'))
            bigram_perplexities_train.append(perplexity_bigram.compute_perplexity(train_data, 'add-k'))
            unigram_perplexities_dev.append(perplexity_unigram.compute_perplexity(dev_data, 'add-k'))
            bigram_perplexities_dev.append(perplexity_bigram.compute_perplexity(dev_data, 'add-k'))
            print("\nk = ", k)
            print(f"Unigram Add-k Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'add-k'):.2f}")
            print(f"Bigram Add-k Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'add-k'):.2f}")
        
        

        unigram_model = LanguageModel(unigram_counter, vocabulary_size)
        bigram_model = LanguageModel(bigram_counter, vocabulary_size)
        perplexity_unigram = Perplexity(unigram_model, preprocessor)
        perplexity_bigram = Perplexity(bigram_model, preprocessor)


        print("\nLaplace Smoothing:")
        unigram_perplexities_train.append(perplexity_unigram.compute_perplexity(train_data, 'laplace'))
        bigram_perplexities_train.append(perplexity_bigram.compute_perplexity(train_data, 'laplace'))
        print(f"Unigram Laplace Smoothing Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'laplace'):.2f}")
        print(f"Bigram Laplace Smoothing Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'laplace'):.2f}")
        unigram_perplexities_dev.append(perplexity_unigram.compute_perplexity(dev_data, 'laplace'))
        bigram_perplexities_dev.append(perplexity_bigram.compute_perplexity(dev_data, 'laplace'))
        k_values = np.append(k_values, 1)
        # print(len(k_values), len(unigram_perplexities), len(bigram_perplexities))

        # if len(k_values) == len(unigram_perplexities) == len(bigram_perplexities):
        plt.clf()
        plt.plot(k_values, unigram_perplexities_train, label='unigram train perplexity plot', color='blue')
        plt.plot(k_values, bigram_perplexities_train, label='bigram train perplexity plot', color='red')
        plt.axvline(x=1, color='green', linestyle='--', label="")
        plt.text(1.02, unigram_perplexities_train[-1], 'Laplace', rotation=90, color='green')
        plt.legend()
        plt.xlabel('k values')
        plt.ylabel('perplexity')
        plt.title('Unigram and Bigram Perplexity vs k values - train data')
        plt.savefig("./unigram_bigram_plots/plot_train_"+str(token_count)+".png")
        
        plt.clf()
        plt.plot(k_values, unigram_perplexities_dev, label='unigram dev perplexity plot', color='#ff7f0e')
        plt.plot(k_values, bigram_perplexities_dev, label='bigram dev perplexity plot', color='#9467bd')
        plt.legend()
        plt.xlabel('k values')
        plt.ylabel('perplexity')
        plt.title('Unigram and Bigram Perplexity vs k values - Dev data')
        plt.axvline(x=1, color='green', linestyle='--', label="")
        plt.text(1.02, unigram_perplexities_train[-1], 'Laplace', rotation=90, color='green')
        plt.savefig("./unigram_bigram_plots/plot_dev_"+str(token_count)+".png")

        print("\nUn-smoothed:")
        print(f"Un-smoothed Unigram Perplexity: {perplexity_unigram.compute_perplexity(dev_data, 'ngram'):.2f}")
        print(f"Un-smoothed Bigram Perplexity: {perplexity_bigram.compute_perplexity(dev_data, 'ngram'):.2f}")
        print("\n\n\n")
    plt.clf()
    plt.plot(missing_token_count, vocab_collect, label='', color='blue')
    plt.xlabel('missing token count')
    plt.ylabel('vocabulary size')
    plt.title('missing token count vs vocabulary sizes')
    plt.savefig("./vocab_plot/vocab_plot.png")


if __name__ == "__main__":
    main()