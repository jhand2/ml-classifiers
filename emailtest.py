import emailparser as parser
from nbtrainer import nbtrainer
from nbclassifier import nbclassifier
import random

root_dir = '/home/jhand/class/cse415/project/enron1/'
ham = parser.parse_directory(root_dir + "ham", "ham")
spam = parser.parse_directory(root_dir + "spam", "spam")
data = ham + spam
random.shuffle(data)

third = len(data) // 3
training_data = data[:third * 2]
test_data = data[third * 2:]

trainer = nbtrainer()
for d in training_data:
    trainer.train(d['attribute'], d['class'])

classifier = nbclassifier(trainer)


def test(name, data):
    classification = classifier.classify(data)
    print('Item ' + name + ' is a ' + classification)

for d in test_data:
    test(d["name"], d["attribute"])
