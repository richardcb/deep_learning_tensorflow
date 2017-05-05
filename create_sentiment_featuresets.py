import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


def create_lexicon(pos, neg):
    lexicon = []
    # take words from data and put them in a lexicon
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[: hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    # lemmatize all words put in lexicon
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # final lexicon with lemmatized words
    l2 = []
    for w in w_counts:
        # keep lexicon efficient by removing uncommon words
        # if a word (w) is seen 1000 or more times, keep as it is a common word
        # if a word is seen 50 or less times, remove as it is uncomoon
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    # shuffle the data in order to not overfit
    random.shuffle(features)
    features = np.array(features)
    test_size = int(test_size * len(features))
    # when using [:, 0] puts all of the zeroeth elements in the x array
    # training data
    train_x = list(features[:, 0][:-test_size])
    train_y = list(features[:, 1][:-test_size])
    # testing data
    test_x = list(features[:, 0][-test_size:])
    test_y = list(features[:, 1][-test_size:])
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)