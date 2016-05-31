#!/bin/python3.5
import sys
import importlib
import time

if len(sys.argv) < 2:
    print("Usage:")
    print("\tclassify.py [model_module]")
    sys.exit(1)
else:
    model = importlib.import_module(sys.argv[1])


def get_classifier(m):
    print("Welcome to a data model of %s." % sys.argv[1])
    print("")
    print("Please choose a classifier.")
    print("Options are: " + str(sorted(list(m.trainers.keys()))))
    return input(">> ")


def get_test_method(m):
    print("")
    print("Please choose a testing method.")
    print("Options are: " + str(sorted(list(m.tests.keys()))))
    test = input(">> ")
    print("")
    return test


def run_test(m, classifier, test):
    s_time = int(round(time.time() * 1000))
    print("Running Test...")
    print("")
    acc = test(m.data, classifier, True)
    end_time = int(round(time.time() * 1000))
    print("Elapsed Time: " + str((end_time - s_time) / 1000))
    print("")
    return acc


def intro():
    options = ["Algorithms", "Bagging", "Comparison"]
    print("Let's explore some machine learing alorithms.")
    print("Which topic would you like to cover? ")
    print("Options are " + str(options))
    return input(">> ")


def compare(m):
    """
    Gets the two algorithms to be compared
    """
    options = sorted(list(m.trainers.keys())) + ["bagging"]
    print("Choose two algorithms to compare.")
    print("Options are: " + str(options))
    print("")
    first = ""
    second = ""
    seen = False
    while first not in options or second not in options:
        if seen:
            print("Not valid algorithms")
            print("")
        first = input("First algorithm: ")
        second = input("Second algorithm: ")
        print("")
        seen = True
    print("")
    return (first.lower(), second.lower())

if __name__ == "__main__":
    m = model
    topic = intro().lower()
    classifier = None
    if topic != "comparison":
        if topic == "algorithms":
            classifier = get_classifier(m)
            test = get_test_method(m)
            training_func = m.trainers[classifier]
            test_func = m.tests[test]
        elif topic == "bagging":
            classifier = topic
            training_func = m.bagging_trainer
            test_func = m.bagging_test
        acc = run_test(m, training_func, test_func) * 100

        triple = (classifier, sys.argv[1], acc)
        print("Accuracy of %s on data model %s: %.2f%%" % triple)
    else:
        algs = compare(m)
        accuracies = []
        print("The data is shuffled and the the holdout method is being used")
        print("to create training/test sets.")
        print("The same training and test sets are used for each algorithm.")
        print("")
        print("Running Tests, this may time a minute or two...")
        for alg in algs:
            s = time.time()
            if alg == "bagging":
                acc = m.tests["holdout"](m.data, m.bagging_trainer, False) * 100
            else:
                acc = m.tests["holdout"](m.data, m.trainers[alg], False) * 100
            e = time.time()
            accuracies.append((alg, acc, e - s))
        print("")
        for a in accuracies:
            triple = (a[0], sys.argv[1], a[1])
            print("Accuracy of %s on data model %s: %.2f%%" % triple)
            print("Elapsed Time: %.2f" % a[2])
            print("")
