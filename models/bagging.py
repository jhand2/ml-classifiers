"""
Jordan Hand, Josh Malters, and Kevin Fong
CSE 415 Spring 2016
Professor: S. Tanimoto
Final Project

Bagging classifier. Uses multiple classifiers and votes on each classification.
"""
import comparison as comp


class bagging(object):
    """
    A machine learning classifier which uses bagging to combine multiple
    trainers to vote on a classification. In theory this provides greater
    accuracy.
    """
    def __init__(self, data, trainers, iterations):
        self.classifiers = []
        self.data = data[:]
        self.indices = set(range(len(data)))
        t_mult = []
        n_trainers = 9
        for i in range(n_trainers):
            t_mult.append(trainers[i % len(trainers)])

        # Trains each classifier on given data set
        for t in t_mult:
            n = len(data)
            split = comp.b_data_split(data, n)
            train = split[0]
            self.indices = self.indices - split[2]
            self.classifiers.append(t(train))

    def classify(self, attributes, output):
        """
        Classifies an item with given attributes. If output is True, prints
        information about each classification.
        """
        votes = {}
        for c in self.classifiers:
            lbl = c.classify(attributes, output)
            if lbl not in votes:
                votes[lbl] = 1
            else:
                votes[lbl] += 1
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        if output:
            print("Chosen label: " + str(sorted_votes[0][0]))
            print("Votes: " + str(list(votes.items())))
        return sorted_votes[0][0]
