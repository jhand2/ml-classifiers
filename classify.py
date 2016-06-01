#!/bin/python3.5
"""
Jordan Hand, Josh Malters, and Kevin Fong
CSE 415 Spring 2016
Professor: S. Tanimoto
Final Project

Main program for this project. Runs interactive dialog to allow users to try out
different machine learning concepts on different data sets.

Only tested with python 3.5

Usage:
    classify.py [data_model]

    available data models: [emails, fruit, chess]
"""
import sys
import os
import importlib
import time

sys.path.append(os.path.abspath('data'))
sys.path.append(os.path.abspath('metrics'))
sys.path.append(os.path.abspath('models'))

# Imports data model
if len(sys.argv) < 2:
    print("Usage:")
    print("\tclassify.py [model_module]")
    sys.exit(1)
else:
    model = importlib.import_module(sys.argv[1])


def intro():
    """
    Dialog intro. Returns the concept chosen by the user.
    """
    options = ["algorithms", "bagging", "comparison"]
    concept = ""
    valid = False
    print("Let's explore some machine learning algorithms.")
    print("Which topic would you like to cover? ")
    while not valid:
        print("Options are " + str(options))
        concept = input(">> ").lower()
        valid = concept in options
        if not valid:
            print("")
            print(concept + " is not an available concept.")
            print("")
    return concept


def get_classifier(m):
    """
    Dialog to get classifier type. Returns the classifier name chosen by the
    user.
    """
    print("Welcome to a data model of %s." % sys.argv[1])
    print("")
    print("Please choose a classifier.")
    options = sorted(list(m.trainers.keys()))
    classifier = ""
    valid = False
    while not valid:
        print("Options are: " + str(options))
        classifier = input(">> ").lower()
        valid = classifier in options
        if not valid:
            print("")
            print(classifier + " is not an available option.")
            print("")
    return classifier


def get_test_method(m):
    print("")
    print("Please choose a testing method.")
    options = sorted(list(m.tests.keys()))
    test = ""
    valid = False
    while not valid:
        print("Options are: " + str(options))
        test = input(">> ").lower()
        print("")
        valid = test in options
        if not valid:
            print(test + " is not an available option.")
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


def compare(m):
    """
    Gets the two algorithms to be compared. Returns algorithm names in a tuple
    """
    options = sorted(list(m.trainers.keys()) + ["bagging"])
    print("Choose two algorithms to compare.")
    first = ""
    second = ""
    valid = False
    while not valid:
        print("Options are: " + str(options))
        print("")
        first = input("First algorithm: ").lower()
        second = input("Second algorithm: ").lower()
        print("")
        valid = (first in options and second in options) or first == second
        if not valid:
            print("One or both of (%s, %s) not valid." % first, second)
            print("Note, both algorithms cannot be the same.")
            print("")
    print("")
    return (first, second)

if __name__ == "__main__":
    """
    Runs this big long dialog. If you want to semi understand this code
    it is probably best to just run the program.
    """

    m = model
    topic = intro()
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
