from enum import Enum
from mslice.plotting.pyplot import CATEGORY_SLICE, CATEGORY_CUT


class IntensityType(Enum):
    SCATTERING_FUNCTION = 1
    CHI = 2
    CHI_MAGNETIC = 3
    D2SIGMA = 4
    SYMMETRISED = 5
    GDOS = 6


class IntensityCache:
    __action_dict = {}
    __method_dict_cut = {}
    __method_dict_slice = {}
    __description_dict = {
        IntensityType.SCATTERING_FUNCTION: "scattering_function",
        IntensityType.CHI: "dynamical_susceptibility",
        IntensityType.CHI_MAGNETIC: "dynamical_susceptibility_magnetic",
        IntensityType.D2SIGMA: "d2sigma",
        IntensityType.SYMMETRISED: "symmetrised",
        IntensityType.GDOS: "gdos",
    }

    def __init__(self):
        pass

    @classmethod
    def _return_method_dict(cls, category):
        if category == CATEGORY_CUT:
            ret_dict = cls.__method_dict_cut
        elif category == CATEGORY_SLICE:
            ret_dict = cls.__method_dict_slice
        else:
            raise ValueError(f"Method invalid: {category}")
        return ret_dict

    @classmethod
    def cache_action(cls, intensity_correction_type, action):
        if intensity_correction_type not in cls.__action_dict:
            cls.__action_dict[intensity_correction_type] = action

    @classmethod
    def get_action(cls, intensity_correction_type):
        if intensity_correction_type in cls.__action_dict:
            return cls.__action_dict[intensity_correction_type]
        else:
            raise KeyError(
                "action related to the specified intensity correctiontype not found"
            )

    @classmethod
    def cache_method(cls, category, intensity_correction_type, method):
        method_dict = cls._return_method_dict(category)
        if intensity_correction_type not in method_dict:
            method_dict[intensity_correction_type] = method

    @classmethod
    def get_method(cls, category, intensity_correction_type):
        method_dict = cls._return_method_dict(category)
        if intensity_correction_type in method_dict:
            return method_dict[intensity_correction_type]
        else:
            raise KeyError("method related to the intensity correction type not found")

    @classmethod
    def get_intensity_type_from_desc(cls, description):
        if description in cls._IntensityCache__description_dict.values():
            return list(cls.__description_dict.keys())[
                list(cls.__description_dict.values()).index(description)
            ]
        else:
            raise ValueError(f"Input intensity type invalid: {description}")

    @classmethod
    def get_desc_from_type(cls, intensity_type):
        if intensity_type in cls.__description_dict.keys():
            return cls.__description_dict[intensity_type]
        else:
            raise ValueError(f"Input intensity type invalid: {intensity_type}")
