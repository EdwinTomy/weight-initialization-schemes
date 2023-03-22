from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import keras
import numpy as np

import random


class LoadDataset:

    @staticmethod
    def load_dataset(dataset, num_classes, input_shape):
        # Model / data parameters
        num_classes = 10
        input_shape = (32, 32, 1)

        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # Initialization Logic
        initialization_set_size = 2048
        indexes = list(range(x_train.shape[0]))
        random_indexes = random.sample(indexes, initialization_set_size)
        init_set_x = x_train[random_indexes]
        init_set_y = y_train[random_indexes]

        return init_set_x, init_set_y, x_train, y_train, x_test, y_test

    @staticmethod
    def load_cifar10():
        return LoadDataset.load_dataset(keras.datasets.cifar10, 10, (32, 32, 1))

    @staticmethod
    def load_mnist():
        return LoadDataset.load_dataset(keras.datasets.mnist, 10, (28, 28, 1))
