class nbtrainer(object):
    def __init__(self):
        super(nbtrainer, self).__init__()
        self.class_count = {}
        self.att_count = {}

    def train(self, attributes, class_name):
        """Adds one count to the given class and adds one to the count of each attribute for that class"""
        self.class_count[class_name] = self.class_count.get(class_name, 0) + 1

        for att in attributes:
            if not att in self.att_count:
                self.att_count[att] = {}
            self.att_count[att][class_name] = self.att_count[att].get(class_name, 0) + 1

    def get_classes(self):
        """Return list of classes"""
        return self.class_count.keys()

    def get_class_count(self, class_name):
        """Return frequency of given class, None if zero"""
        return self.class_count.get(class_name, None)

    def get_data_count(self):
        """Return total number of data"""
        return sum(self.class_count.values())

    def get_att_count(self, att, class_name):
        """Return frequency of attribute for given class, None if zero"""
        try:
            return self.att_count[att][class_name]
        except:
            return None

