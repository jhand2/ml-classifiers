import time


class knnclassifier(object):
    def __init__(self, data, keys, k):
        self.data = data
        self.feat_key = keys[0]
        self.lbl_key = keys[1]
        self.k = k

    def __hueristic_distance(self, instance1, instance2):
        '''
        Returns the calculated similarity between any two given data instances.
        In this case returns the number of similar words between two emails.
        '''
        att1 = instance1
        att2 = instance2
        same = set.intersection(att1, att2)
        return len(same)

    def get_neighbors(self, d, k):
        '''
        Returns the k most similar instances for a given unseen instance d
        '''
        distances = []
        for record in self.data:
            dist = self.__hueristic_distance(d, record[self.feat_key])
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
            s_time = time.time() * 1000
            response = n[self.lbl_key]
            e_time = time.time() * 1000
            print("Elapsed time for one response = " + str(e_time - s_time))
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sorted_v = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        return sorted_v[0][0]
