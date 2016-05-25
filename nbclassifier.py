import math


class nbclassifier(object):

    def __init__(self, trainer):
        super(nbclassifier, self).__init__()
        self.data = trainer
        self.nonzero = .0000001

    def classify(self, attributes):
        """Returns classification based on highest probability"""
        classes = self.data.get_classes()
        probs = {}

        # P(att|className)
        for class_name in classes:
            atts_probs =\
                [self.get_att_prob(att, class_name) for att in attributes]

            # P(attributes|class_name) =
            # P(att1|class_name) * P(att2|class_name) * ... * P(attn|class_name)
            total_att_prob = 0
            if len(atts_probs) > 0:
                total_att_prob = 1
                for prob in atts_probs:
                    total_att_prob = total_att_prob + prob

            # P(className|att)
            probs[class_name] = total_att_prob + self.get_prior_prob(class_name)

        # Returns className with highest probability
        classification = ""
        best = self.nonzero
        for class_name in classes:
            if probs[class_name] > best:
                best = probs[class_name]
                classification = class_name
        return classification

    def get_prior_prob(self, cls_name):
        """Return P(cls_name)"""
        p = self.data.get_class_count(cls_name) / self.data.get_data_count()
        return math.log(p)

    def get_att_prob(self, att, class_name):
        """Return the log (base e) of P(att|class_name)"""
        total = self.data.get_class_count(class_name)
        att_count = self.data.get_att_count(att, class_name)

        if att_count is None:
            return self.nonzero

        return math.log(att_count / total)
