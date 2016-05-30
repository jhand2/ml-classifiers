import emailparser as parser
from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
from knnclassifier import knnclassifier
from bagging import bagging
import comparison as comp
import os
import numpy as np

root_dir = os.path.abspath('enron1/')
ham = parser.parse_directory(root_dir + "/ham", "ham")
spam = parser.parse_directory(root_dir + "/spam", "spam")
data = np.concatenate((spam, ham))
np.random.shuffle(data)

att = "attribute"
lbl = "class"
n = "name"
keys = (att, lbl, n)


def train_nb(training_data):
    """
    Trains email data on a naive bayes classifier
    """
    trainer = nbtrainer()
    trainer.train(training_data, keys)
    classifier = nbclassifier(trainer)
    return classifier


def train_knn(training_data):
    """
    Trains email data on a k nearest neighbors classifier
    """
    return knnclassifier(training_data, keys, 3)


def classify_test(classifier, test_data):
    """
    Classifies each data point, prints the name and classification of eveything
    in test_data.
    """
    for d in test_data:
        test(d["name"], d["attribute"], classifier)


def test(name, data, classifier):
    """
    Classifies a data point data with the given classifier and prints its
    name and classification.
    """
    classification = classifier.classify(data)
    print('Item ' + name + ' is a ' + classification)


def bagging_trainer(training_data):
    return bagging(training_data, [train_nb, train_knn], 10)


def bagging_test(d, t, output):
    return comp.bagging_test(d, t, output, keys)


# Classification techniques available for this formulation
trainers = {
    "naive_bayes": train_nb,
    "knn": train_knn,
}

# Testing methods available for this formulation
tests = {
    "holdout": lambda d, t, o, k=keys: comp.holdout_test(d, t, o, k),
    "bootstrap": lambda d, t, o, k=keys: comp.bootstrap_test(d, t, o, k)
}
