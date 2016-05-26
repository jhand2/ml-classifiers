import emailparser as parser
from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
import random
import comparison as comp

root_dir = '/Users/Josh/Documents/cse 415/ml-classifiers/enron1/'
ham = parser.parse_directory(root_dir + "ham", "ham")
spam = parser.parse_directory(root_dir + "spam", "spam")
data = ham + spam
random.shuffle(data)


# REDUNDANT FILE SO I CAN TINKER WITH THINGS

def train_nb(training_data):
    trainer = nbtrainer()
    for d in training_data:
        trainer.train(d['attribute'], d['class'])

    classifier = nbclassifier(trainer)
    return classifier


def classify_test(classifier, test_data):
    for d in test_data:
        test(d["name"], d["attribute"], classifier)


def test(name, data, classifier):
    classification = classifier.classify(data)
    print('Item ' + name + ' is a ' + classification)


att = "attribute"
lbl = "class"
n = "name"
keys = (att, lbl, n)
trainer = train_nb

trainers = {
    "naive_bayes": train_nb
}

tests = {
    "holdout": lambda d, t, k=keys: comp.holdout_test(d, t, k)
}

if __name__ == "__main__":
    nb_acc = tests["holdout"](data, trainers["naive_bayes"])
    print("Naive Bayes Accuracy: " + str(nb_acc))
