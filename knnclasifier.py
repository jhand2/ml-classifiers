import math
import random
import emailparser as parser
import operator

root_dir = '/Users/Josh/Documents/cse 415/ml-classifiers/enron1/'
ham = parser.parse_directory(root_dir + "ham", "ham")
spam = parser.parse_directory(root_dir + "spam", "spam")
data = ham + spam
random.shuffle(data)

def hueristicDistance(instance1, instance2):
    attr1 = instance1['attribute']
    attr2 = instance2['attribute']
    dif = [x for x in attr1 in attr2]

    return len(dif)

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    for set in trainingSet:
        for x in range(len(trainingSet)):
            dist = hueristicDistance(testInstance, trainingSet[x])
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors


if __name__ == "__main__":
    trainSet = data
    testInstance = ham[0]
    testSet = []
    k = 1
    neighbors = getNeighbors(trainSet, testInstance)
    print(neighbors)
