from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from tensorflow.keras.datasets import cifar10
import keras
from initialization_techniques.scheme2_init import *
from initialization_techniques.scheme2_init import WeightInitScheme2Params

import random


def main():

    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 1)

    # Load the data and split it between train and test sets
    (x_train_c, y_train_c), (x_test_c, y_test_c) = keras.datasets.cifar10.load_data()

    # Scale images to the [0, 1] range
    x_train_c = x_train_c.astype("float32") / 255
    x_test_c = x_test_c.astype("float32") / 255
    # Make sure images have shape (32, 32, 1)
    x_train_c = np.expand_dims(x_train_c, -1)
    x_test_c = np.expand_dims(x_test_c, -1)
    print("x_train shape:", x_train_c.shape)
    print(x_train_c.shape[0], "train samples")
    print(x_test_c.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train_c = keras.utils.to_categorical(y_train_c, num_classes)
    y_test_c = keras.utils.to_categorical(y_test_c, num_classes)

    conv2_model = keras.Sequential(
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

    batch_size = 60
    epochs = 1#5
    opt = keras.optimizers.SGD(learning_rate=0.01)
    conv2_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Initialization Logic
    initialization_set_size = 2048
    indexes = list(range(x_train_c.shape[0]))
    random_indexes = random.sample(indexes, initialization_set_size)
    init_set_x = x_train_c[random_indexes]
    init_set_y = y_train_c[random_indexes]

    # - Defining the parameters of the initialization technique for ReLU layers
    layers_init_params = WeightInitScheme2Params(
        batch=init_set_x,
        verbose=True,
    )

    # - Run initialization process
    model = WeightInitScheme2.initialize(conv2_model, layers_init_params)

    # Compute loss and accuracy on initialization dataset
    loss_and_acc = model.test_on_batch(init_set_x, init_set_y)
    init_set_loss, init_set_acc = loss_and_acc[0], loss_and_acc[1]
    print("init_set_loss:", init_set_loss, "init_set_acc", init_set_acc)

    loss_and_acc = model.test_on_batch(x_test_c, y_test_c)
    initial_val_loss, initial_val_acc = loss_and_acc[0], loss_and_acc[1]
    print("initial_val_loss:", initial_val_loss, "initial_val_acc", initial_val_acc)

    train_history = model.fit(x_train_c, y_train_c, batch_size=batch_size, epochs=epochs,
                              validation_split=0.1, shuffle=True)

    val_loss = train_history.history['val_loss']
    val_acc = train_history.history['val_accuracy']

    val_loss.insert(0, initial_val_loss)
    val_acc.insert(0, initial_val_acc)
    print("Loss: ", val_loss)
    print("Accuracy: ", val_acc)

    # Baseline
    train_history = conv2_model.fit(x_train_c, y_train_c, batch_size=batch_size, epochs=epochs,
                              validation_split=0.1, shuffle=True)


    val_loss = train_history.history['val_loss']
    val_acc = train_history.history['val_accuracy']

    print("Loss: ", val_loss)
    print("Accuracy: ", val_acc)

if __name__ == '__main__':
    main()
