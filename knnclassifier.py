from time import time as t


class knnclassifier(object):
    def __init__(self, data, keys, k):
        self.data = data
        self.feat_key = keys[0]
        self.lbl_key = keys[1]
        self.k = k

    def distance(self, d_atts, existing):
        '''
        Returns the calculated similarity between any two given data instances.
        In this case returns the number of similar words between two emails.
        '''
        att1 = existing[self.feat_key]
        att2 = d_atts
        same = att1 & att2
        return len(same)

    def get_neighbors(self, d, k):
        '''
        Returns the k most similar instances for a given unseen instance d
        '''
        distances = []
        for record in self.data:
            dist = self.distance(d, record)
            distances.append((record, dist))
        distances.sort(key=lambda x: x[1], reverse=True)
        neighbors = []

        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def classify(self, d):
        '''
        Returns the majority voted response from a number of neighbors.
        '''
        neighbors = self.get_neighbors(d, self.k)
        classVotes = {}
        for n in neighbors:
            response = n[self.lbl_key]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sorted_v = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        return sorted_v[0][0]
