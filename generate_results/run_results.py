from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from initialization_techniques.initialization_techniques_manager import initializationTechniqueOptionsName
import initialization_techniques.base_scheme_init as base_scheme_init
from tensorflow.keras.datasets import cifar10
import keras
import numpy as np

from initialization_techniques.scheme5_init import WeightInitScheme5Params
from models.cifar10_models import CIFARModels

import random


def run_model(model_type, initial_weights, initialization, init_set_x, init_set_y, x_test, y_test,
              x_train, y_train, opt=keras.optimizers.SGD(learning_rate=0.01), batch_size=60, epochs=10):
    print("################################", initialization, " ##################################")
    model = CIFARModels.create_model(model_type)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.set_weights(initial_weights)

    # - Defining the parameters of the initialization technique for ReLU layers
    params = base_scheme_init.WeightInitSchemeParams(batch=init_set_x, verbose=True)

    # - Run initialization process
    model = base_scheme_init.WeightInitScheme.initialize(initialization, model, params)

    loss_and_acc = model.test_on_batch(init_set_x, init_set_y)
    init_set_loss, init_set_acc = loss_and_acc[0], loss_and_acc[1]
    print("init_set_loss:", init_set_loss, "init_set_acc", init_set_acc)

    loss_and_acc = model.test_on_batch(x_test, y_test)
    initial_val_loss, initial_val_acc = loss_and_acc[0], loss_and_acc[1]
    print("initial_val_loss:", initial_val_loss, "initial_val_acc", initial_val_acc)

    train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                              validation_split=0.1)

    val_loss = train_history.history['val_loss']
    val_acc = train_history.history['val_accuracy']

    val_loss.insert(0, initial_val_loss)
    val_acc.insert(0, initial_val_acc)
    print("Loss: ", val_loss)
    print("Accuracy: ", val_acc)
    return val_loss, val_acc


def main():
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

    batch_size = 60
    epochs = 10
    opt = keras.optimizers.SGD(learning_rate=0.01)

    # Initialization Logic
    initialization_set_size = 2048
    indexes = list(range(x_train.shape[0]))
    random_indexes = random.sample(indexes, initialization_set_size)
    init_set_x = x_train[random_indexes]
    init_set_y = y_train[random_indexes]

    # Model list
    how_many = 10
    model_list = ["cifar2", "cifar2_no_cl", "cifar4", "cifar4_no_cl", "cifar6", "cifar6_no_cl"]
    storage = np.zeros((how_many, len(model_list), 4, 2, epochs + 1))

    for i in range(how_many):
        for j, model_type in enumerate(model_list):
            model = CIFARModels.create_model(model_type)
            initial_weights = model.get_weights()

            # Baseline
            val_loss, val_acc = run_model(model_type, initial_weights, initializationTechniqueOptionsName.BASELINE,
                                          init_set_x, init_set_y, x_test, y_test, x_train, y_train)
            storage[i, j, 0, 0, :] = val_loss
            storage[i, j, 0, 1, :] = val_acc

            # Scheme 2
            val_loss, val_acc = run_model(model_type, initial_weights, initializationTechniqueOptionsName.SCHEME2,
                                          init_set_x, init_set_y, x_test, y_test, x_train, y_train)
            storage[i, j, 1, 0, :] = val_loss
            storage[i, j, 1, 1, :] = val_acc

            # Scheme 3
            val_loss, val_acc = run_model(model_type, initial_weights, initializationTechniqueOptionsName.SCHEME3,
                                          init_set_x, init_set_y, x_test, y_test, x_train, y_train)
            storage[i, j, 2, 0, :] = val_loss
            storage[i, j, 2, 1, :] = val_acc

            np.save("results/cifar_results", storage)

    np.save("results/cifar_results", storage)


if __name__ == '__main__':
    main()
