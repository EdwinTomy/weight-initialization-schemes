from __future__ import print_function
import numpy as np
import math
from utils.utils import Utils
import tensorflow as tf
from tensorflow.keras.layers import Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D


class WeightInitScheme3Params:
    def __init__(self, batch, verbose):
        self.batch = batch
        self.verbose = verbose


class WeightInitScheme3:

    @staticmethod
    def initialize(model, params):

        batch = params.batch
        verbose = params.verbose if params.verbose is not None else True
        layers_initialized = 0

        if verbose:
            print("------- Scheme 3 - Initialization Process Started ------- ")

        for i in range(len(model.layers)):
            layer = model.layers[i]
            classes_to_consider = (Dense)

            if not isinstance(layer, classes_to_consider):
                if verbose:
                    print("Scheme3 - skipping " + layer.name + ' - not in the list of classes to be initialized')
                continue

            # Get output of layers from using the initialization set
            layer_output = Utils.get_layer_activations(model, layer, batch)
            layer_output = layer_output.reshape((-1, layer_output.shape[-1]))

            # Obtain weights and biases
            weights_and_biases = layer.get_weights()
            last_dim = weights_and_biases[0].shape[-1]
            new_weights = weights_and_biases[0].reshape((-1, last_dim))
            new_biases = weights_and_biases[1]

            h1_s_std = np.std(layer_output)
            if h1_s_std != 0:
                new_weights /= h1_s_std
            else:
                print("Warning, h1_s_std=0")
            if new_weights.std() > 1:
                print("Warning, new_weights.std() > 1")
            weight_dev = min(1, new_weights.std())
            new_weights = np.random.choice([0, 1, -1], new_weights.shape,
                                           p=[1 - (weight_dev ** 2), (weight_dev ** 2) / 2,
                                              (weight_dev ** 2) / 2])
            new_biases = np.zeros(new_biases.shape)

            new_weights = np.reshape(new_weights, weights_and_biases[0].shape)
            weights_and_biases[0] = new_weights
            weights_and_biases[1] = new_biases
            layer.set_weights(weights_and_biases)

            # Print some statistics about the weights/biases and the layer's activations
            if verbose:
                weights_and_biases = layer.get_weights()
                new_weights = weights_and_biases[0].reshape((-1, last_dim))
                new_biases = weights_and_biases[1]
                layer_output = Utils.get_layer_activations(model, layer, batch)
                layer_output = layer_output.reshape((-1))
                new_weights = new_weights.reshape((-1, new_weights.shape[-1]))
                new_biases = new_biases.reshape((-1, new_biases.shape[-1]))

                print("------- Scheme 3 - Layer initialized: " + layer.name + " ------- ")
                print("Weights -- Std: ", np.std(new_weights), " Mean: ", np.mean(new_weights), " Max: ",
                      np.max(new_weights), " Min: ", np.min(new_weights))
                print("Biases -- Std: ", np.std(new_biases), " Mean: ", np.mean(new_biases), " Max: ",
                      np.max(new_biases), " Min: ", np.min(new_biases))
                print("Layer activations' std: ", np.std(layer_output, axis=0))
                print("Layer activations' mean: ", np.mean(layer_output, axis=0))

            layers_initialized += 1

        if verbose:
            print("------- Scheme 3 - DONE - total layers initialized ", layers_initialized, "------- ")

        return model
