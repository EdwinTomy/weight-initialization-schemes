from __future__ import print_function

import keras

from initialization_techniques.initialization_techniques_manager import InitializationTechniqueOptionsName
from utils.load_datasets import LoadDataset
import numpy as np
from models.cifar10_models import CIFARModels
import run_models as r

def main():
    # Load the data
    init_set_x, init_set_y, x_train, y_train, x_test, y_test = LoadDataset.load_cifar10()
    batch_size = 60
    epochs = 10
    opt = keras.optimizers.SGD(learning_rate=0.01)

    # Model list
    how_many = 10
    model_list = ["cifar2", "cifar2_no_cl", "cifar4", "cifar4_no_cl", "cifar6", "cifar6_no_cl"]
    storage = np.zeros((how_many, len(model_list), 4, 2, epochs + 1))

    for i in range(how_many):
        for j, model_type in enumerate(model_list):
            model = CIFARModels.create_model(model_type)
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            initial_weights = model.get_weights()

            # Baseline
            val_loss, val_acc = r.run_model(model_type, initial_weights, InitializationTechniqueOptionsName.BASELINE,
                                            init_set_x, init_set_y, x_test, y_test, x_train, y_train)
            storage[i, j, 0, 0, :] = val_loss
            storage[i, j, 0, 1, :] = val_acc

            # Scheme 2
            val_loss, val_acc = r.run_model(model_type, initial_weights, InitializationTechniqueOptionsName.SCHEME2,
                                            init_set_x, init_set_y, x_test, y_test, x_train, y_train)
            storage[i, j, 1, 0, :] = val_loss
            storage[i, j, 1, 1, :] = val_acc

            # Scheme 3
            val_loss, val_acc = r.run_model(model_type, initial_weights, InitializationTechniqueOptionsName.SCHEME3,
                                            init_set_x, init_set_y, x_test, y_test, x_train, y_train)
            storage[i, j, 2, 0, :] = val_loss
            storage[i, j, 2, 1, :] = val_acc

            np.save("results/cifar_results", storage)

    np.save("results/cifar_results", storage)


if __name__ == '__main__':
    main()
