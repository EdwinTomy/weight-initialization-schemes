from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from enum import Enum
from tensorflow.keras.datasets import cifar10
import keras


class MNISTModels:

    @staticmethod
    def create_model(model_type):
        num_classes = 10
        if model_type == "mnist":
            return keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dense(300, activation="relu"),
                    layers.Dense(100, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "mnist_no_cl":
            return keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(300, activation="relu"),
                    layers.Dense(100, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

