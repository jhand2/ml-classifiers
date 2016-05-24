from nbtrainer import nbtrainer
from nbclassifier import nbclassifier

training_data = [{'attribute': ['not long', 'not yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'not yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'not yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'not yellow'], 'class': 'other'},
                {'attribute': ['not long', 'not yellow'], 'class': 'other'},
                {'attribute': ['not long', 'not yellow'], 'class': 'other'},
                {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'yellow'], 'class': 'lemon'},
                {'attribute': ['not long', 'yellow'], 'class': 'other'},
                {'attribute': ['long', 'not yellow'], 'class': 'banana'},
                {'attribute': ['long', 'not yellow'], 'class': 'banana'},
                {'attribute': ['long', 'not yellow'], 'class': 'banana'},
                {'attribute': ['long', 'not yellow'], 'class': 'other'},
                {'attribute': ['long', 'not yellow'], 'class': 'other'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'},
                {'attribute': ['long', 'yellow'], 'class': 'banana'}]

trainer = nbtrainer()
for data in training_data:
    trainer.train(data['attribute'], data['class'])

classifier = nbclassifier(trainer)

def test(data):
    classification = classifier.classify(data)
    print('Item that is: [' + ', '.join(data) + '] is a ' + classification)

test1 = ['not long', 'not yellow']
test2 = ['not long', 'yellow']
test3 = ['long', 'not yellow']
test4 = ['long', 'yellow']
test(test1)
test(test2)
test(test3)
test(test4)