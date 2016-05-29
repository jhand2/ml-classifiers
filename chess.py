import emailparser as parser
from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
from knnclassifier import knnclassifier
import random
import comparison as comp
import os

root_dir = os.path.abspath('enron2/')
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
data = draw + zero + one + two + three + four + five + six + seven + eight + nine + ten + \
    eleven + twelve + thirteen + fourteen + fifteen + sixteen
random.shuffle(data)


def train_nb(training_data):
    trainer = nbtrainer()
    for d in training_data:
        trainer.train(d['attribute'], d['class'])

    classifier = nbclassifier(trainer)
    return classifier


def train_knn(training_data):
    return knnclassifier(training_data, 3)


def classify_test(classifier, test_data):
    for d in test_data:
        test(d["name"], d["attribute"], classifier)


def test(name, data, classifier):
    classification = classifier.classify(data)
    print('Board ' + name + ' is a ' + classification + ' (move(s) win)')


att = "attribute"
lbl = "class"
n = "name"
keys = (att, lbl, n)

trainers = {
    "naive_bayes": train_nb,
    "knn": train_knn
}

tests = {
    "holdout": lambda d, t, k=keys: comp.holdout_test(d, t, k)
}