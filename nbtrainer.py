class nbtrainer(object):
    def __init__(self):
        super(nbtrainer, self).__init__()
        self.class_count = {}
        self.att_count = {}

    def train(self, data, keys):
        """
        Adds one count to the given class and adds one to the count of
        each attribute for that class

        keys (triple): A triple of keys to access pieces of the data

            feature_key: Used to get the features of an individual data point
            label_key: Used to get the label of an individual data point
            name_key: Used to get the name of an individual data point
        """
        for d in data:
            self.class_count[d[keys[1]]] =\
                self.class_count.get(d[keys[1]], 0) + 1

            for att in d[keys[0]]:
                if att not in self.att_count:
                    self.att_count[att] = {}
                self.att_count[att][d[keys[1]]] =\
                    self.att_count[att].get(d[keys[1]], 0) + 1

    def get_classes(self):
        """Return list of classes"""
        return self.class_count.keys()

    def get_class_count(self, class_name):
        """Return frequency of given class"""
        return self.class_count.get(class_name, 0)

    def get_data_count(self):
        """Return total number of data"""
        return sum(self.class_count.values())

    def get_att_count(self, att, class_name):
        """Return frequency of attribute for given class, None if zero"""
        try:
            return self.att_count[att][class_name]
        except:
            return 0
