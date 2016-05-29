import comparison as comp
import random as r


class bagging(object):
    def __init__(self, data, trainers, iterations):
        self.classifiers = []
        self.data = data[:]
        self.indices = set(range(len(data)))
        t_mult = []
        for i in range(5):
            t_mult.append(r.choice(trainers))
        for t in t_mult:
            n = len(data)
            split = comp.b_data_split(data, n)
            train = split[0]
            self.indices = self.indices - split[2]
            self.classifiers.append(t(train))

    def classify(self, attributes):
        votes = {}
        for c in self.classifiers:
            lbl = c.classify(attributes)
            if lbl not in votes:
                votes[lbl] = 1
            else:
                votes[lbl] += 1
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        print("Chosen label: " + str(sorted_votes[0][0]))
        print("Votes: " + str(list(votes.items())))
        return sorted_votes[0][0]

    def get_test_data(self):
        test_data = []
        for i in self.indices:
            test_data.append(self.data[i])
        return test_data
