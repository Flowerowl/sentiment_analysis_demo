#coding: utf-8
import itertools

import jieba
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.metrics import BigramAssocMeasures
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist, ConditionalFreqDist


def create_word_scores(pos_words, neg_words, pos_tag, neg_tag):
    pos_words = list(itertools.chain(*pos_words))
    neg_words = list(itertools.chain(*neg_words))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos_words:
        word_fd[word] += 1
        cond_word_fd[pos_tag][word] += 1
    for word in neg_words:
        word_fd[word] += 1
        cond_word_fd[neg_tag][word] += 1

    pos_word_count = cond_word_fd[pos_tag].N()
    neg_word_count = cond_word_fd[neg_tag].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd[pos_tag][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd[neg_tag][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])

    return best_words


def best_features(review, best_words):
    return dict([(feature, True) for feature in review if feature in best_words])


def build_classifier_score(train_set, test_set, classifier):
    data, tag = zip(*test_set)
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)
    pred = classifier.classify_many(data)

    return accuracy_score(tag, pred)


if __name__ == "__main__":
    review_num = 5000
    with open('pos_reviews.txt', 'r') as f:
        pos_review = map(lambda data: list(jieba.cut(data)), f.readlines())[: review_num]
    with open('neg_reviews.txt', 'r') as f:
        neg_review = map(lambda data: list(jieba.cut(data)), f.readlines())[: review_num]

    word_scores = create_word_scores(pos_review, neg_review, 'pos', 'neg')
    best_words = find_best_words(word_scores, review_num)
    pos_features = []
    for review in pos_review:
        pos_features.append([best_features(review, best_words), 'pos'])
    neg_features = []
    for review in neg_review:
        neg_features.append([best_features(review, best_words), 'neg'])

    pos_train_data, pos_test_data = train_test_split(pos_features, test_size=0.3)
    neg_train_data, neg_test_data = train_test_split(neg_features, test_size=0.3)
    train_set = pos_train_data + neg_train_data
    test_set = pos_test_data + neg_test_data

    classifer_dict = {
        'BernoulliNB': BernoulliNB(),
        'MultinomialNB': MultinomialNB(),
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        'NuSVC': NuSVC()
    }

    for classifer_name, classifer_func in classifer_dict.items():
        print classifer_name + "'s accuracy is %f" % build_classifier_score(train_set, test_set, classifer_func)
