import random as r
import numpy as np


def holdout_test(data, training_func, output, keys):
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
    n = (len(data) // 10) * 9
    training_data, test_data = holdout_split(data, n)
    classifier = training_func(training_data)

    correct = 0
    total = len(test_data)
    classified = {}
    for d in test_data:
        label = classifier.classify(d[keys[0]], output)
        classified[d[keys[2]]] = label
        if label == d[keys[1]]:
            correct += 1
    return correct / total


def holdout_split(data, n):
    data_cpy = data[:]
    r.shuffle(data_cpy)
    training_data = np.array(data_cpy[:n])
    test_data = np.array(data_cpy[n:])
    return (training_data, test_data)


def bootstrap_test(data, training_func, output, keys):
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
    n = (len(data) // 10) * 9
    training, test_data, _ = b_data_split(data, n)
    correct = 0
    total = len(test_data)
    classified = {}
    classifier = training_func(training)

    for d in test_data:
        label = classifier.classify(d[keys[0]], output)
        classified[d[keys[2]]] = label
        if label == d[keys[1]]:
            correct += 1

    return correct / total


def bagging_test(data, training_func, output, keys):
    n = (len(data) // 10) * 9
    training, test_data = holdout_split(data, n)
    print("Classifying " + str(len(test_data)) + " data points.")
    print("")
    correct = 0
    classified = {}
    classifier = training_func(training)

    total = len(test_data)

    for d in test_data:
        label = classifier.classify(d[keys[0]], output)
        classified[d[keys[2]]] = label
        print("Actual Label: " + str(d[keys[1]]))
        print("")
        if label == d[keys[1]]:
            correct += 1

    return correct / total


def b_data_split(data, n):
    """
    Splits the data using the bootstrap method
    """
    # n = (len(data) // 10) * 9
    # n = len(data)
    training = []
    indices = set()
    data_range = range(len(data))
    for i in range(n):
        sample = r.choice(data_range)
        training.append(data[sample])
        indices.add(sample)
    training = np.array(training)

    test_data = []
    for i in data_range:
        if i not in indices:
            test_data.append(data[i])
    return (training, test_data, indices)
