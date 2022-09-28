from enum import Enum
from mslice.plotting.pyplot import CATEGORY_SLICE, CATEGORY_CUT

ACTION, METHOD = 1, 2

class IntensityType(Enum):
    SCATTERING_FUNCTION = 1
    CHI = 2
    CHI_MAGNETIC = 3
    D2_SIGMA = 4
    SYMMETRISED = 5
    GDOS = 6


class IntensityCache:
    __action_dict_cut = {}
    __method_dict_cut = {}
    __action_dict_slice = {}
    __method_dict_slice = {}
    __identifier_dict = {}

    def __init__(self):
        pass

    @classmethod
    def _return_category_dict(cls, category, identifier):
        if identifier == ACTION:
            if category == CATEGORY_CUT:
                ret_dict = cls.__action_dict_cut
            elif category == CATEGORY_SLICE:
                ret_dict = cls.__action_dict_slice
        elif identifier == METHOD:
            if category == CATEGORY_CUT:
                ret_dict = cls.__method_dict_cut
            elif category == CATEGORY_SLICE:
                ret_dict = cls.__method_dict_slice
        else:
            raise ValueError(f"Category invalid: {category}")
        return ret_dict

    @classmethod
    def cache_action(cls, category, ax, intensity_correction_type, action):
        action_dict = cls._return_category_dict(category, ACTION)
        if ax not in action_dict:
            action_dict[ax] = {intensity_correction_type: action}
        else:
            action_dict[ax][intensity_correction_type] = action

    @classmethod
    def get_action(cls, category, ax, intensity_correction_type):
        action_dict = cls._return_category_dict(category, ACTION)
        if ax in action_dict and intensity_correction_type in action_dict[ax]:
            return action_dict[ax][intensity_correction_type]
        else:
            raise KeyError("action related to the specified axis and intensity correction"
                           "type not found")

    @classmethod
    def trigger_action(cls, category, ax, intensity_correction_type):
        action = cls.get_action(category, ax, intensity_correction_type)
        action.trigger()

    @classmethod
    def cache_method(cls, category, intensity_correction_type, method):
        method_dict = cls._return_category_dict(category, METHOD)
        if intensity_correction_type not in method_dict:
            method_dict[intensity_correction_type] = method

    @classmethod
    def get_method(cls, category, intensity_correction_type):
        method_dict = cls._return_category_dict(category, METHOD)
        if intensity_correction_type in method_dict:
            return method_dict[intensity_correction_type]
        else:
            raise KeyError("method related to the specified intensity correction type not found")

    @classmethod
    def get_intensity_type_from_desc(cls, description):
        if description == "scattering_function":
            return IntensityType.SCATTERING_FUNCTION
        elif description == "dynamical_susceptibility":
            return IntensityType.CHI
        elif description == "dynamical_susceptibility_magnetic":
            return IntensityType.CHI_MAGNETIC
        elif description == "d2sigma":
            return IntensityType.D2_SIGMA
        elif description == "symmetrised":
            return IntensityType.SYMMETRISED
        elif description == "gdos":
            return IntensityType.GDOS
        else:
            raise ValueError(f"Input intensity type invalid: {description}")

    @classmethod
    def get_desc_from_type(cls, intensity_type):
        if intensity_type == IntensityType.SCATTERING_FUNCTION:
            return "scattering_function"
        elif intensity_type == IntensityType.CHI:
            return "dynamical_susceptibility"
        elif intensity_type == IntensityType.CHI_MAGNETIC:
            return "dynamical_susceptibility_magnetic"
        elif intensity_type == IntensityType.D2_SIGMA:
            return "d2sigma"
        elif intensity_type == IntensityType.SYMMETRISED:
            return "symmetrised"
        elif intensity_type == IntensityType.GDOS:
            return "gdos"
        else:
            raise ValueError(f"Input intensity type invalid: {intensity_type}")

    @classmethod
    def remove_from_cache(cls, category, key):
        dicts = (cls._return_category_dict(category, ACTION), cls._return_category_dict(category, METHOD))
        for d in dicts:
            if key in d.keys():
                d.pop(key)
