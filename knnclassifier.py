class knnclassifier(object):
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def __hueristic_distance(self, instance1, instance2):
        '''
        Returns the calculated similarity between any two given data instances.
        In this case returns the number of similar words between two emails.
        '''
        attr1 = instance1
        attr2 = instance2
        dif = [x for x in attr2 if x in attr1]
        return len(dif)

    def get_neighbors(self, d, k):
        '''
        Returns the k most similar instances for a given unseen instance d
        '''
        distances = []
        for record in self.data:
            dist = self.__hueristic_distance(d, record)
            distances.append((record, dist))
        distances.sort(key=lambda x: x[1], reverse=True)
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def classify(self, d):
        '''
        Returns the majority voted response from a number of neighbors. It
        assumes the class is the 'class' attribute for each neighbor dictionary
        '''
        neighbors = self.get_neighbors(d, self.k)
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x]['class']
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sorted_v = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
        return sorted_v[0][0]
