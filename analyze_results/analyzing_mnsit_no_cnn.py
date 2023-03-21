# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def main():
    result_mnist = np.load("/Users/edwintomy/PycharmProjects/weight-initialization-schemes/results"
                           "/mnist_nocnn1_results.npy")

    num_tests, num_, num_models, _, epochs = result_mnist.shape

    # plot
    fig, ax = plt.subplots()

    print(result_mnist.shape)
    print(result_mnist)

    x = np.linspace(0, 10, 11)

    for i in range(10):
        y = result_mnist[i, 0, 2, 1, :]
        ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(0, 6), xticks=np.arange(0, 6),
           )
    plt.title("History of Loss on Lenet trained on MNIST: Scheme 3")

    plt.show()

    result_mnist.shape

    # plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 6)

    for i in range(10):
        y = result_mnist[i, 1, 1, 1, :]
        ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(0, 9), xticks=np.arange(1, 8),
           ylim=(0, 1), yticks=np.arange(1, 8))

    plt.show()

    # plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 11)

    y0 = np.mean(result_mnist[[1, 2, 5, 6, 7, 8, 9], 0, 0, 1, :], axis=0)
    y1 = np.mean(result_mnist[:, 0, 1, 1, :], axis=0)
    y2 = np.mean(result_mnist[[1, 2, 5, 6, 7, 8, 9], 0, 2, 1, :], axis=0)

    ax.plot(x, y0, linewidth=2.0, label="baseline")
    ax.plot(x, y1, linewidth=2.0, label="scheme ")
    ax.plot(x, y2, linewidth=2.0, label="scheme 3")

    ax.set(xlim=(0, 6), xticks=np.arange(1, 8),
           ylim=(0.5, 1), yticks=np.arange(0, 2))

    leg = plt.legend(loc='upper center')
    plt.title("History of Accuracy on Lenet trained on MNIST")
    plt.show()

    # plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 11)

    y0 = np.mean(result_mnist[:, 0, 0, 1, :], axis=0)
    y1 = np.mean(result_mnist[:, 0, 1, 1, :], axis=0)
    y2 = np.mean(result_mnist[:, 0, 2, 1, :], axis=0)

    ax.plot(x, y0, linewidth=2.0, label="baseline")
    ax.plot(x, y1, linewidth=2.0, label="scheme 2/3")
    ax.plot(x, y2, linewidth=2.0, label="scheme 6")

    ax.set(xlim=(0, 1), xticks=np.arange(0, 11),
           ylim=(0.9, 1), yticks=np.arange(90, 100) / 100)

    leg = plt.legend(loc='upper center')

    plt.show()

    # plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 11)

    y0 = np.mean(result_mnist[:, 0, 0, 1, :], axis=0)
    y1 = np.mean(result_mnist[:, 0, 1, 1, :], axis=0)
    y2 = np.mean(result_mnist[:, 0, 2, 1, :], axis=0)

    ax.plot(x, y0, linewidth=2.0, label="baseline")
    ax.plot(x, y1, linewidth=2.0, label="scheme 2/3")
    ax.plot(x, y2, linewidth=2.0, label="scheme 6")

    ax.set(xlim=(0, 1), xticks=np.arange(0, 11),
           ylim=(0, 0.31), yticks=np.arange(0, 30) / 100)

    leg = plt.legend(loc='upper center')

    plt.show()

    # plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 6)

    y0 = np.mean(result_mnist[:, 0, 0, 0, :], axis=0)
    y1 = np.mean(result_mnist[:, 0, 1, 0, :], axis=0)
    y2 = np.mean(result_mnist[:, 0, 2, 0, :], axis=0)

    ax.plot(x, y0, linewidth=2.0, label="baseline")
    ax.plot(x, y1, linewidth=2.0, label="scheme 2")
    ax.plot(x, y2, linewidth=2.0, label="scheme 3")

    ax.set(xlim=(0, 9), xticks=np.arange(1, 8),
           ylim=(0, 1), yticks=np.arange(1, 8))

    leg = plt.legend(loc='upper center')

    plt.show()

    # plot
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 6)

    y0 = np.mean(result_mnist[:, 1, 0, 0, :], axis=0)
    y1 = np.mean(result_mnist[:, 1, 1, 0, :], axis=0)
    y2 = np.mean(result_mnist[:, 1, 2, 0, :], axis=0)

    ax.plot(x, y0, linewidth=2.0, label="baseline")
    ax.plot(x, y1, linewidth=2.0, label="scheme 2")
    ax.plot(x, y2, linewidth=2.0, label="scheme 3")

    ax.set(xlim=(0, 9), xticks=np.arange(1, 8),
           ylim=(0, 1), yticks=np.arange(1, 8))

    leg = plt.legend(loc='upper center')

    plt.show()


if __name__ == '__main__':
    main()
