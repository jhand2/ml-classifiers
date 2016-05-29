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

if __name__ == "__main__":
    m = model
    print("Welcome to a data model of %s." % sys.argv[1])
    print("")
    print("Please choose a classifier.")
    print("Options are: " + str(list(m.trainers.keys())))
    classifier = input(">> ")
    print("")
    print("Please choose a testing method.")
    print("Options are: " + str(list(m.tests.keys())))
    test = input(">> ")
    print("")

    s_time = int(round(time.time() * 1000))
    print("Running Test...")
    print("")
    acc = m.tests[test](m.data, m.trainers[classifier])
    end_time = int(round(time.time() * 1000))
    acc_str = "Accuracy of %s on data model %s: %s" %\
        (classifier, sys.argv[1], acc)
    print(acc_str)
    print("Elapsed Time: " + str((end_time - s_time) / 1000))
