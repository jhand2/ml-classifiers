import emailparser as parser
from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
from knnclassifier import knnclassifier
import random
import comparison as comp
import os

root_dir = os.path.abspath('enron1/')
ham = parser.parse_directory(root_dir + "/ham", "ham")
spam = parser.parse_directory(root_dir + "/spam", "spam")
data = ham + spam
random.shuffle(data)


def train_nb(training_data):
    trainer = nbtrainer()
    for d in training_data:
        trainer.train(d['attribute'], d['class'])

    classifier = nbclassifier(trainer)
    return classifier


def train_knn(training_data):
    return knnclassifier(training_data, 3)


def classify_test(classifier, test_data):
    for d in test_data:
        test(d["name"], d["attribute"], classifier)


def test(name, data, classifier):
    classification = classifier.classify(data)
    print('Item ' + name + ' is a ' + classification)


att = "attribute"
lbl = "class"
n = "name"
keys = (att, lbl, n)

trainers = {
    "naive_bayes": train_nb,
    "knn": train_knn
}

tests = {
    "holdout": lambda d, t, k=keys: comp.holdout_test(d, t, k)
}
