import sys
import importlib

if len(sys.argv) < 2:
    print("Usage:")
    print("\tclassify.py [model_module]")
    sys.exit(1)
else:
    model = importlib.import_module(sys.argv[1])

if __name__ == "__main__":
    print("Welcome to a data model of %s." % sys.argv[1])
    print("Please choose a classifier.")
    print("")
    print("Options are: " + str(list(model.trainers.keys())))
    classifier = input(">> ")
    m = model
    acc = m.tests["holdout"](m.data, m.trainers["naive_bayes"])
    acc_str = "Accuracy of %s on data model %s: %s" %\
        (classifier, sys.argv[1], acc)
    print(acc_str)
