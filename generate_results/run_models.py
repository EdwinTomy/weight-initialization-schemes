from __future__ import print_function
import initialization_techniques.base_scheme_init as base_scheme_init
import initialization_techniques.initialization_techniques_manager as init_manager
import keras
from models.cifar10_models import CIFARModels
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


def run_model(model_type, initial_weights, initialization, init_set_x, init_set_y, x_test, y_test,
              x_train, y_train, opt=keras.optimizers.SGD(learning_rate=0.01), batch_size=60, epochs=10, num_classes=10):
    print("################################", initialization, " ##################################")
    model = CIFARModels.create_model(model_type)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.set_weights(initial_weights)

    # - Defining the parameters of the initialization technique for ReLU layers
    params = base_scheme_init.WeightInitSchemeParams(batch=init_set_x, verbose=True)

    # - Run initialization process
    model = init_manager.InitializationTechniqueManager.get_initialization_technique(initialization, model, params)
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
