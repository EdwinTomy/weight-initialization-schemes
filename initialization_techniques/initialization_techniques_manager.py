from enum import Enum
from initialization_techniques.he_init import WeightInitHe
from initialization_techniques.glorot_init import WeightInitGlorot
from initialization_techniques.lsuv_init import WeightInitLSUV
from initialization_techniques.scheme1_init import WeightInitScheme1
from initialization_techniques.scheme4_init import WeightInitScheme4
from initialization_techniques.scheme2_init import WeightInitScheme2
from initialization_techniques.scheme3_init import WeightInitScheme3


class InitializationTechniqueOptions(Enum):
    HE = 1
    GLOROT = 2
    LSUV = 3
    SCHEME1 = 4
    SCHEME4 = 5
    SCHEME2 = 6
    SCHEME3 = 7

class InitializationTechniqueManager:
    @staticmethod
    def get_initialization_technique(init_technique_option):

        weight_init_dictionary = {
            InitializationTechniqueOptions.HE: WeightInitHe,
            InitializationTechniqueOptions.GLOROT: WeightInitGlorot,
            InitializationTechniqueOptions.LSUV: WeightInitLSUV,
            InitializationTechniqueOptions.SCHEME1: WeightInitScheme1,
            InitializationTechniqueOptions.SCHEME4: WeightInitScheme4,
            InitializationTechniqueOptions.SCHEME2: WeightInitScheme2,
            InitializationTechniqueOptions.SCHEME3: WeightInitScheme3,
        }

        weight_init_class = weight_init_dictionary.get(init_technique_option, WeightInitHe)

        return weight_init_class


def test_init_techniques_manager():
    weight_init_class = \
        InitializationTechniqueManager.get_initialization_technique(InitializationTechniqueOptions.SCHEME4)

    print(weight_init_class)


if __name__ == "__main__":
    test_init_techniques_manager()
