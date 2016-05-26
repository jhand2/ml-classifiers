#!/bin/python3.5

from nbtrainer import nbtrainer
from nbclassifier import nbclassifier

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

for d in data:
    d["name"] = ", ".join(d["attribute"])


def train_nb(training_data):
    trainer = nbtrainer()
    for data in training_data:
        trainer.train(data['attribute'], data['class'])

    classifier = nbclassifier(trainer)
    return classifier


def test(data, classifier):
    classification = classifier.classify(data)
    print('Item that is: [' + ', '.join(data) + '] is a ' + classification)

keys = ("attribute", "class", "name")

trainers = {
    "naive_bayes": train_nb
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
