#!/bin/python3.5

from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
import comparison as comp
from knnclassifier import knnclassifier
from bagging import bagging

data = [
    {'attribute': ['not long', 'not yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'not yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'not yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'not yellow'], 'class': 'other'},
    {'attribute': ['not long', 'not yellow'], 'class': 'other'},
    {'attribute': ['not long', 'not yellow'], 'class': 'other'},
    {'attribute': ['not long', 'not yellow'], 'class': 'other'},
    {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
    {'attribute': ['not long', 'yellow'], 'class': 'other'},
    {'attribute': ['long', 'not yellow'], 'class': 'banana'},
    {'attribute': ['long', 'not yellow'], 'class': 'banana'},
    {'attribute': ['long', 'not yellow'], 'class': 'banana'},
    {'attribute': ['long', 'not yellow'], 'class': 'other'},
    {'attribute': ['long', 'not yellow'], 'class': 'other'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'},
    {'attribute': ['long', 'yellow'], 'class': 'banana'}
]

keys = ("attribute", "class", "name")
# random.shuffle(data)

for d in data:
    d["name"] = ", ".join(d["attribute"])
    d["attribute"] = set(d["attribute"])


def train_nb(training_data):
    trainer = nbtrainer()
    trainer.train(training_data, keys)

    classifier = nbclassifier(trainer)
    return classifier


def train_knn(training_data):
    return knnclassifier(training_data, keys, 3)


def test(data, classifier):
    classification = classifier.classify(data)
    print('Item that is: [' + ', '.join(data) + '] is a ' + classification)


def bagging_trainer(training_data):
    return bagging(training_data, [train_nb, train_knn], 10)


def bagging_test(d, t, output):
    return comp.bagging_test(d, t, output, keys)

trainers = {
    "naive_bayes": train_nb,
    "knn": train_knn
}

tests = {
    "holdout": lambda d, t, o, k=keys: comp.holdout_test(d, t, o, k),
    "bootstrap": lambda d, t, o, k=keys: comp.bootstrap_test(d, t, o, k)
}

if __name__ == "__main__":
    test1 = ['not long', 'not yellow']
    test2 = ['not long', 'yellow']
    test3 = ['long', 'not yellow']
    test4 = ['long', 'yellow']
    test(test1)
    test(test2)
    test(test3)
    test(test4)
