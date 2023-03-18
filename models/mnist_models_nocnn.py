from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from enum import Enum
from tensorflow.keras.datasets import cifar10
import keras
from initialization_techniques.scheme3_init import *
from initialization_techniques.scheme3_init import WeightInitScheme3Params
from initialization_techniques.scheme6_init import *
from initialization_techniques.scheme6_init import WeightInitScheme6Params


class MNISTModels:

    @staticmethod
    def create_model(model_type):
        num_classes = 10
        if model_type == "mnist_nocnn":
            return keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "mnist_nocnn_4":
            return keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(4, activation="relu"),
                    layers.Dense(4, activation="relu"),
                    layers.Dense(4, activation="relu"),
                    layers.Dense(4, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "mnist_nocnn_8":
            return keras.Sequential(
                [
                    layers.Input(shape=(28, 28, 1)),
                    layers.Flatten(),
                    layers.Dense(8, activation="relu"),
                    layers.Dense(8, activation="relu"),
                    layers.Dense(8, activation="relu"),
                    layers.Dense(8, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )



