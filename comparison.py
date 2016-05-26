def holdout_test(data, training_func, keys):
    """
    Uses the holdout method on the given data to test accuracy.

    params:
        data (list): A list of data objects

        keys (triple): A triple of keys to access pieces of the data

            feature_key: The key used to get the label of an individual data
                         point

            label_key: The key used to get the attributes of an individual data
                       point

            name_key: The key used to geth the name of an individual data point

        training_func (function): A function that can take a subset of data to
                                  train a classifier. Should return a classifier
    """
    third = len(data) // 3
    training_data = data[:third * 2]
    test_data = data[third * 2:]
    classifier = training_func(training_data)

    correct = 0
    total = len(test_data)
    classified = {}
    for d in test_data:
        label = classifier.classify(d[keys[0]])
        classified[d[keys[2]]] = label
        if label == d[keys[1]]:
            correct += 1
    return correct / total
