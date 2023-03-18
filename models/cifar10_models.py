from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from enum import Enum
from tensorflow.keras.datasets import cifar10
import keras
from initialization_techniques.scheme2_init import *
from initialization_techniques.scheme2_init import WeightInitScheme2Params
from initialization_techniques.scheme3_init import *
from initialization_techniques.scheme3_init import WeightInitScheme3Params
from initialization_techniques.scheme5_init import *
from initialization_techniques.scheme5_init import WeightInitScheme5Params


class CIFARModels:

    @staticmethod
    def create_model(model_type):
        num_classes = 10
        if model_type == "cifar2":
            return keras.Sequential(
                [
                    layers.Input(shape=(32, 32, 3)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "cifar2_no_cl":
            return keras.Sequential(
                [
                    layers.Input(shape=(32, 32, 3)),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "cifar4":
            return keras.Sequential(
                [
                    layers.Input(shape=(32, 32, 3)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "cifar4_no_cl":
            return keras.Sequential(
                [
                    layers.Input(shape=(32, 32, 3)),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "cifar6":
            return keras.Sequential(
                [
                    layers.Input(shape=(32, 32, 3)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same'),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

        if model_type == "cifar6_no_cl":
            return keras.Sequential(
                [
                    layers.Input(shape=(32, 32, 3)),
                    layers.Flatten(),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )
