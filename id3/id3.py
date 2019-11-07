import numpy as np
import pandas as pd
from pprint import pprint


class DecisionTree(object):
    def __init__(self, data, features, target):
        self.data = data
        self.features = features
        self.target = target
        # self.attributes = attributes
        # self.labels = labels
        # self.root = None
        # self.entropy = self.getEntropy([x for x in range(len(self.labels))])

    def get_entropy(self, column):
        elements, counts = np.unique(column, return_counts=True)
        entropy = np.sum(
            [(-counts[i]/np.sum(counts))
             * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
        return entropy

    def get_info_gain(self, data, attribute, target):
        total_entropy = self.get_entropy(data[target])
        values, counts = np.unique(
            data[attribute], return_counts=True)

        igains = []
        for i in range(len(values)):
            igain = counts[i]/np.sum(counts) * \
                self.get_entropy(data.where(
                    data[attribute] == values[i]).dropna()[target])
            igains.append(igain)

        weighted_entropy = np.sum(igains)
        return total_entropy - weighted_entropy

    def id3(self, data, originaldata, features, target_attribute_name, parent_node_class=None):
        # determine root node
        # If all target_values have the same value, return this value
        data_dimensions = np.unique(data[target_attribute_name])
        if len(data_dimensions) <= 1:
            return data_dimensions[0]
        elif len(data) == 0:
            return np.unique(originaldata[target_attribute_name])[
                np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
        elif len(features) == 0:
            return parent_node_class
        else:
            parent_node_class = data_dimensions[np.argmax(
                np.unique(data[target_attribute_name], return_counts=True)[1])]

            # Return the information gain values for the features in the dataset
            item_values = [self.get_info_gain(data, feature, target_attribute_name)
                           for feature in features]
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]

            tree = {best_feature: {}}
            features = [i for i in features if i != best_feature]

            for value in np.unique(data[best_feature]):
                value = value
                sub_data = data.where(data[best_feature] == value).dropna()
                subtree = self.id3(sub_data, data, features,
                                   target_attribute_name, parent_node_class)
                tree[best_feature][value] = subtree

            return tree

    def use_id3(self):
        features = [i for i in self.features if i != self.target.name]
        return self.id3(self.data, self.data, features, self.target.name)


def main():
    # data_headers = ['engine', 'turbo', 'weight', 'fueleco', 'fast']
    data = pd.read_csv("id3_data.csv")
    features = data.columns
    print(features)
    target = data.fast
    decison_tree = DecisionTree(data, features, target)
    tree = decison_tree.use_id3()
    pprint(tree)


if __name__ == "__main__":
    main()
