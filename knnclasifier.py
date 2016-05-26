import math
import random
import emailparser as parser
import operator

root_dir = '/Users/Josh/Documents/cse 415/ml-classifiers/enron1/'
ham = parser.parse_directory(root_dir + "ham", "ham")
spam = parser.parse_directory(root_dir + "spam", "spam")
data = ham + spam
random.shuffle(data)


def hueristic_distance(instance1, instance2):
    '''Returns the calculated similarity between any two given data instances.
    In this case returns the number of similar words between two emails.'''
    attr1 = instance1['attribute']
    attr2 = instance2['attribute']
    dif = [x for x in attr2 if x in attr1]
    return len(dif)


def get_neighbors(trainingSet, testInstance, k):
    '''Returns the k most similar instances for a given unseen instance.'''
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = hueristic_distance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist))
    distances.sort(key=lambda x: x[1], reverse=True)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    '''Returns the majority voted response from a number of neighbors. It
    assumes the class is the 'class' attribute for each neighbor dictionary'''
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]['class']
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    print(classVotes.items())
    sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
    return sortedVotes[0][0]


if __name__ == "__main__":
    trainSet = data
    testInstance = random.choice(ham)
    testSet = []
    k = 5  # Important that k is odd so there are no ties
    neighbors = get_neighbors(trainSet, testInstance, k)
    response = get_response(neighbors)
    print(response)
