import random as r
import numpy as np


def holdout_test(data, training_func, keys):
    """
    Uses the holdout method on the given data to test accuracy.

    params:
        data (list): A list of data objects

        keys (triple): A triple of keys to access pieces of the data

            feature_key: Used to get the attributes of an individual data point
            label_key: Used to get the features of an individual data point
            name_key: Used to get the name of an individual data point

        training_func (function): A function that can take a subset of data to
                                  train a classifier. Should return a classifier
    """
    data_cpy = data[:]
    r.shuffle(data_cpy)
    third = len(data_cpy) // 3
    training_data = np.array(data_cpy[:third * 2])
    test_data = np.array(data_cpy[third * 2:])
    classifier = training_func(training_data)

    correct = 0
    total = len(test_data)
    classified = {}
    for d in test_data:
        label = classifier.classify(d[keys[0]])
        classified[d[keys[2]]] = label
        if label == d[keys[1]]:
            correct += 1
    return correct / total


def bootstrap_test(data, training_func, keys):
    """
    Uses the bootstrap method on the given data to test accuracy.

    params:
        data (list): A list of data objects

        keys (triple): A triple of keys to access pieces of the data

            feature_key: Used to get the features of an individual data point
            label_key: Used to get the label of an individual data point
            name_key: Used to get the name of an individual data point

        training_func (function): A function that can take a subset of data to
                                  train a classifier. Should return a classifier
    """
    n = (len(data) // 3) * 2
    training = []
    indices = set()
    data_range = range(len(data))
    for i in range(n):
        sample = r.choice(data_range)
        training.append(data[sample])
        indices.add(sample)
    training = np.array(training)

    test = []
    for i in data_range:
        if i not in indices:
            test.append(data[i])
    test = np.array(test)

    classifier = training_func(training)

    correct = 0
    total = len(test)
    classified = {}
    for d in test:
        label = classifier.classify(d[keys[0]])
        classified[d[keys[2]]] = label
        if label == d[keys[1]]:
            correct += 1
    return correct / total
