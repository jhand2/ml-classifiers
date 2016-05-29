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
    print("Options are: " + str(list(m.trainers.keys())))
    return input(">> ")


def get_test_method(m):
    print("")
    print("Please choose a testing method.")
    print("Options are: " + str(list(m.tests.keys())))
    test = input(">> ")
    print("")
    return test


def run_test(m, classifier, test):
    s_time = int(round(time.time() * 1000))
    print("Running Test...")
    print("")
    acc = test(m.data, classifier)
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

if __name__ == "__main__":
    m = model
    topic = intro().lower()
    # classifier = None
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
