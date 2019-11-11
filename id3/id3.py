from pprint import pprint

import pandas as pd

from decision_tree import DecisionTree


def main():
    # data_headers = ['engine', 'turbo', 'weight', 'fueleco', 'fast']
    training_data = pd.read_csv("training_data.csv")
    testing_data = pd.read_csv("testing_data.csv")

    features = training_data.columns
    print(features)

    target = training_data.fast
    decison_tree = DecisionTree(training_data, features, target)

    tree = decison_tree.use_id3()
    pprint(tree)

    decison_tree.test(testing_data, tree)

    decison_tree.draw_graph(tree)


if __name__ == "__main__":
    main()
