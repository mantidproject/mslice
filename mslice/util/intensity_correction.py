from enum import Enum


class IntensityType(Enum):
    SCATTERING_FUNCTION = 1
    CHI = 2
    CHI_MAGNETIC = 3
    D2_SIGMA = 4
    SYMMETRISED = 5
    GDOS = 6


class IntensityCache:
    __action_dict = {}
    __method_dict = {}
    __identifier_dict = {}

    def __init__(self):
        pass

    @classmethod
    def cache_action(cls, ax, intensity_correction_type, action):
        if ax not in cls.__action_dict:
            cls.__action_dict[ax] = {intensity_correction_type: action}
        else:
            cls.__action_dict[ax][intensity_correction_type] = action

    @classmethod
    def get_action(cls, ax, intensity_correction_type):
        if ax in cls.__action_dict and intensity_correction_type in cls.__action_dict[ax]:
            return cls.__action_dict[ax][intensity_correction_type]
        else:
            raise KeyError("action related to the specified axis and intensity correction"
                           "type not found")

    @classmethod
    def trigger_action(cls, ax, intensity_correction_type):
        action = cls.get_action(ax, intensity_correction_type)
        action.trigger()

    @classmethod
    def cache_method(cls, intensity_correction_type, method):
        if intensity_correction_type not in cls.__method_dict:
            cls.__method_dict[intensity_correction_type] = method

    @classmethod
    def get_method(cls, intensity_correction_type):
        if intensity_correction_type in cls.__method_dict:
            return cls.__method_dict[intensity_correction_type]
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
        else:
            raise ValueError(f"Input intensity type invalid: {intensity_type}")