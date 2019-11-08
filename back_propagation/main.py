import numpy as np

from neural_network import NeuralNetwork

# input
x = np.array([
    [30, 40, 50],
    [40, 50, 20],
    [50, 20, 15],
    [20, 15, 60],
    [15, 60, 70],
    [60, 70, 50]
], dtype=np.float64)

# Expected output
y = np.array([20, 15, 60, 70, 50, 40], dtype=np.float64)


def main():
    NN = NeuralNetwork(x, y)
    # for i in range(1000):
    #     if i % 100 == 0:
    #         print("for iteration # " + str(i) + "\n")
    #         print("Input : \n" + str(x))
    #         print("Actual Output: \n" + str(y))
    #         print("Predicted Output: \n" + str(NN.feed_foward()))
    #         # mean sum squared loss
    #         print("Loss: \n" + str(np.mean(np.square(y - NN.feed_foward()))))
    #         print("\n")

    print(NN.feed_foward())
    print(NN.y)


if __name__ == "__main__":
    main()
