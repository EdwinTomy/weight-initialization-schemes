from enum import Enum
import numpy as np
from initialization_techniques.base_scheme_init import WeightInitScheme
from utils.utils import Utils
from tensorflow.keras.layers import Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D


class InitializationTechniqueOptions(Enum):
    SCHEME2 = 2
    SCHEME3 = 3
    SCHEME5 = 5
    SCHEME6 = 6


class InitializationTechniqueOptionsName(Enum):
    SCHEME2 = 'SCHEME 2'
    SCHEME3 = 'SCHEME 3'
    SCHEME5 = 'SCHEME 5'
    SCHEME6 = 'SCHEME 6'


class InitializationTechniqueManager:

    @staticmethod
    def weight_init_scheme_2_and_3(model, layer, batch, new_weights):
        layer_output = Utils.get_layer_activations(model, layer, batch)
        layer_output = layer_output.reshape((-1, layer_output.shape[-1]))
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

        return new_weights

    @staticmethod
    def weight_init_scheme_5(new_weights):
        new_weights = np.random.choice([0, 1, -1], new_weights.shape,
                                       p=[0.10, 0.45, 0.45])

        return new_weights

    @staticmethod
    def weight_init_scheme_6(new_weights):
        mu_pos, mu_neg, sigma = 1, -1, 0.1  # mean and standard deviation
        pos_weights = np.random.normal(mu_pos, sigma, new_weights.size // 2)
        neg_weights = np.random.normal(mu_neg, sigma, (new_weights.size + 1) // 2)
        pol_weights = np.concatenate((pos_weights, neg_weights))
        new_weights = pol_weights.reshape(new_weights.shape)
        np.random.shuffle(new_weights)

        return new_weights

    @staticmethod
    def get_initialization_technique(init_technique_option_name, model, params):

        convolutional_classes = (Dense, Convolution2D, Convolution1D, Convolution3D, Conv1D, Conv2D, Conv3D)
        non_convolutional_classes = (Dense)

        weight_init_dictionary = {
            InitializationTechniqueOptions.SCHEME2: WeightInitScheme.initialize(
                model, params, InitializationTechniqueOptionsName.weight_init_scheme_2_and_3,
                InitializationTechniqueOptionsName.SCHEME2, convolutional_classes),

            InitializationTechniqueOptions.SCHEME3: WeightInitScheme.initialize(
                model, params, InitializationTechniqueOptionsName.weight_init_scheme_2_and_3,
                InitializationTechniqueOptionsName.SCHEME3, non_convolutional_classes),

            InitializationTechniqueOptions.SCHEME5: WeightInitScheme.initialize(
                model, params, InitializationTechniqueOptionsName.weight_init_scheme_5,
                InitializationTechniqueOptionsName.SCHEME5, non_convolutional_classes),

            InitializationTechniqueOptions.SCHEME6: WeightInitScheme.initialize(
                model, params, InitializationTechniqueOptionsName.weight_init_scheme_6,
                InitializationTechniqueOptionsName.SCHEME6, non_convolutional_classes),
        }

        model = weight_init_dictionary.get(init_technique_option_name)
        return model
