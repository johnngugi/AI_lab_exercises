from collections import Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


class DecisionTree:
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

    def id3(self, data, originaldata, features, target_attribute_name="fast", parent_node_class=None):

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
                sub_data = data.where(data[best_feature] == value).dropna()
                subtree = self.id3(sub_data, data, features,
                                   target_attribute_name, parent_node_class)
                tree[best_feature][value] = subtree

            return tree

    def use_id3(self):
        features = [i for i in self.features if i != self.target.name]
        return self.id3(self.data, self.data, features, self.target.name)

    def predict(self, query, tree, default="no"):
        for key in list(query.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][query[key]]
                except:
                    return default

                result = tree[key][query[key]]

                if isinstance(result, dict):
                    return self.predict(query, result)
                else:
                    return result

    def test(self, data, tree):
        # Create new query instances by simply removing the target feature column from the original dataset and
        # convert it to a dictionary
        queries = data.iloc[:, :-1].to_dict(orient="records")

        # Create a empty DataFrame in whose columns the prediction of the tree are stored
        predicted = pd.DataFrame(columns=["predicted"])

        # Calculate the prediction accuracy
        for i in range(len(data)):
            predicted.loc[i, "predicted"] = self.predict(
                queries[i], tree, "yes")

        # print(predicted["predicted"])
        self.report(data["fast"], predicted["predicted"])
        return predicted

    def report(self, actual, predicted):
        matrix = confusion_matrix(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        classification = classification_report(actual, predicted)

        print("\nResults of test: \n")
        print("Confusion matrix: ")
        print(matrix, "\n")
        print("Accuracy: ", accuracy, "\n")
        print("Classification Report: \n")
        print(classification)

    def draw_graph(self, tree):
        g = nx.Graph()

        q = list(tree.items())

        while q:
            v, d = q.pop()
            for nv, nd in d.items():
                g.add_edge(v, nv)
                if isinstance(nd, Mapping):
                    q.append((nv, nd))

        nx.draw(g, with_labels=True)
        plt.show()
