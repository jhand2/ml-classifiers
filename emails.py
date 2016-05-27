import emailparser as parser
from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
from knnclassifier import knnclassifier
import comparison as comp
import os
import numpy as np

root_dir = os.path.abspath('enron1/')
ham = parser.parse_directory(root_dir + "/ham", "ham")
spam = parser.parse_directory(root_dir + "/spam", "spam")
data = np.concatenate((spam, ham))

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


# Classification techniques available for this formulation
trainers = {
    "naive_bayes": train_nb,
    "knn": train_knn
}

# Testing methods available for this formulation
tests = {
    "holdout": lambda d, t, k=keys: comp.holdout_test(d, t, k),
    "bootstrap": lambda d, t, k=keys: comp.bootstrap_test(d, t, k)
}
