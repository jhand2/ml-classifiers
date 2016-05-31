"""
Jordan Hand, Josh Malters, and Kevin Fong
CSE 415 Spring 2016
Professor: S. Tanimoto

A data formulation of chess end positions. The data set has various records
of the location of each king and the White rook in a chess game. The boards
are classified by how many moves it takes for either player to win the game.
"""
import chessparser as parser
from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
from knnclassifier import knnclassifier
from bagging import bagging
import random
import comparison as comp
import os

# Parse each category
root_dir = os.path.abspath('data/chessdata/')
draw = parser.parse_directory(root_dir + "/draw", "draw")
zero = parser.parse_directory(root_dir + "/zero", "zero")
one = parser.parse_directory(root_dir + "/one", "one")
two = parser.parse_directory(root_dir + "/two", "two")
three = parser.parse_directory(root_dir + "/three", "three")
four = parser.parse_directory(root_dir + "/four", "four")
five = parser.parse_directory(root_dir + "/five", "five")
six = parser.parse_directory(root_dir + "/six", "six")
seven = parser.parse_directory(root_dir + "/seven", "seven")
eight = parser.parse_directory(root_dir + "/eight", "eight")
nine = parser.parse_directory(root_dir + "/nine", "nine")
ten = parser.parse_directory(root_dir + "/ten", "ten")
eleven = parser.parse_directory(root_dir + "/eleven", "eleven")
twelve = parser.parse_directory(root_dir + "/twelve", "twelve")
thirteen = parser.parse_directory(root_dir + "/thirteen", "thirteen")
fourteen = parser.parse_directory(root_dir + "/fourteen", "fourteen")
fifteen = parser.parse_directory(root_dir + "/fifteen", "fifteen")
sixteen = parser.parse_directory(root_dir + "/sixteen", "sixteen")

data = draw + zero + one + two + three + four + five + six + seven + eight +\
       nine + ten + eleven + twelve + thirteen + fourteen + fifteen + sixteen
random.shuffle(data)

# Keys used to get the features, label, and name of each data point
att = "attribute"
lbl = "class"
n = "name"
keys = (att, lbl, n)


def train_nb(training_data):
    """
    Trains email data on a naive bayes classifier
    """
    trainer = nbtrainer()
    trainer.train(training_data, keys)
    classifier = nbclassifier(trainer)
    return classifier


def train_knn(training_data):
    """
    Trains email data on a k nearest neighbors classifier
    """
    return knnclassifier(training_data, keys, 3)


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


# A couple test functions. Not for use in production
def classify_test(classifier, test_data):
    for d in test_data:
        test(d["name"], d["attribute"], classifier)


def test(name, data, classifier):
    classification = classifier.classify(data)
    print('Board ' + name + ' is a ' + classification + ' (move(s) win)')
