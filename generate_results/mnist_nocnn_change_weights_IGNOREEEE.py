from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import cifar10
import keras
from initialization_techniques.scheme3_init import *
from initialization_techniques.scheme3_init import WeightInitScheme3Params
from initialization_techniques.scheme5_init import *
from initialization_techniques.scheme5_init import WeightInitScheme5Params
from models.mnist_models_nocnn import MNISTModels

import random

###### Ignore this file #####################
def main():
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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
    epochs = 5
    opt = keras.optimizers.SGD(learning_rate=0.01)

    # Initialization Logic
    initialization_set_size = 2048
    indexes = list(range(x_train.shape[0]))
    random_indexes = random.sample(indexes, initialization_set_size)
    init_set_x = x_train[random_indexes]
    init_set_y = y_train[random_indexes]

    # Model list
    how_many = 1
    model_list = ["mnist_nocnn_4"]
    storage = np.zeros((how_many, len(model_list), 1, 2, epochs + 1))

    for i in range(how_many):
        for j, model_type in enumerate(model_list):
            print("################################ Baseline ##################################")
            model = MNISTModels.create_model(model_type)
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            initial_weights = model.get_weights()

            ################################## Scheme 6 ##################################
            print("################################ Scheme 6 ##################################")
            model = MNISTModels.create_model(model_type)
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            model.set_weights(initial_weights)

            # - Defining the parameters of the initialization technique for ReLU layers
            layers_init_params = WeightInitScheme5Params(
                batch=init_set_x,
                verbose=True,
            )

            # - Run initialization process
            model = WeightInitScheme5.initialize(model, layers_init_params)

            init_ones = model.get_weights()
            init_ones_weights = init_ones[0]

            # Compute loss and accuracy on initialization dataset
            loss_and_acc = model.test_on_batch(init_set_x, init_set_y)
            init_set_loss, init_set_acc = loss_and_acc[0], loss_and_acc[1]
            print("init_set_loss:", init_set_loss, "init_set_acc", init_set_acc)

            loss_and_acc = model.test_on_batch(x_test, y_test)
            initial_val_loss, initial_val_acc = loss_and_acc[0], loss_and_acc[1]
            print("initial_val_loss:", initial_val_loss, "initial_val_acc", initial_val_acc)

            train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                      validation_split=0.1, shuffle=True)

            val_loss = train_history.history['val_loss']
            val_acc = train_history.history['val_accuracy']

            val_loss.insert(0, initial_val_loss)
            val_acc.insert(0, initial_val_acc)
            print("Loss: ", val_loss)
            print("Accuracy: ", val_acc)

            after = model.get_weights()[0]
            after_true = abs(after) > 1
            how_much = after_true[after_true == True]
            print("locoooo", how_much.size, after_true.size)
            print("after", after)
            print("after", np.amax(after))
            print("after_true", after_true)

            storage[i, j, 0, 0, :] = val_loss
            storage[i, j, 0, 1, :] = val_acc

            ################################## Scheme 6 ##################################
            print("################################ next ##################################")

            init_ones_weights[after_true == False] = init_ones_weights[after_true == False] * -1
            init_ones[0] = init_ones_weights
            model.set_weights(init_ones)

            # Compute loss and accuracy on initialization dataset
            loss_and_acc = model.test_on_batch(init_set_x, init_set_y)
            init_set_loss, init_set_acc = loss_and_acc[0], loss_and_acc[1]
            print("init_set_loss:", init_set_loss, "init_set_acc", init_set_acc)

            loss_and_acc = model.test_on_batch(x_test, y_test)
            initial_val_loss, initial_val_acc = loss_and_acc[0], loss_and_acc[1]
            print("initial_val_loss:", initial_val_loss, "initial_val_acc", initial_val_acc)

            train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                      validation_split=0.1, shuffle=True)

            val_loss = train_history.history['val_loss']
            val_acc = train_history.history['val_accuracy']

            val_loss.insert(0, initial_val_loss)
            val_acc.insert(0, initial_val_acc)
            print("Loss: ", val_loss)
            print("Accuracy: ", val_acc)

            after = model.get_weights()[0]
            after_true = abs(after) > 1
            how_much = after_true[after_true == True]
            print("locoooo", how_much.size, after_true.size)
            print("after", after)
            print("after", np.amax(after))
            print("after_true", after_true)

            storage[i, j, 0, 0, :] = val_loss
            storage[i, j, 0, 1, :] = val_acc

    np.save("results/mnist_nocnn2_results", storage)


if __name__ == '__main__':
    main()
