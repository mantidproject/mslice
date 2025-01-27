"""Module to create axis and legend labels"""

from mslice.models.units import EnergyUnits

CUT_INTENSITY_LABEL = "Signal/#Events"
recoil_labels = {1: "Hydrogen", 2: "Deuterium", 4: "Helium"}

# unit labels for projections
MOD_Q_LABEL = "|Q|"
THETA_LABEL = "2Theta"
DELTA_E_LABEL = "DeltaE"
TWOTHETA_UNITS = ("Degrees", "2Theta")
MOMENTUM_UNITS = ("MomentumTransfer", "|Q|", "Angstrom^-1")


def is_twotheta(unit):
    return any([unit in val for val in TWOTHETA_UNITS])


def is_momentum(unit):
    return any([unit in val for val in MOMENTUM_UNITS])


def is_energy(unit):
    return unit == DELTA_E_LABEL


def are_units_equivalent(unit_lhs, unit_rhs):
    if is_twotheta(unit_lhs) and is_twotheta(unit_rhs):
        return True
    elif is_momentum(unit_lhs) and is_momentum(unit_rhs):
        return True
    elif is_energy(unit_lhs) and is_energy(unit_rhs):
        return True
    else:
        return False


def get_recoil_key(label):
    for key, value in recoil_labels.items():
        if value == label:
            return key
    return label


def get_display_name(axis):
    if "DeltaE" in axis.units:
        return EnergyUnits(axis.e_unit).label()
    elif is_momentum(axis.units):
        # Matplotlib 1.3 doesn't handle LaTeX very well. Sometimes no legend appears if we use LaTeX
        return r"$|Q|$ ($\mathrm{\AA}^{-1}$)"
    elif is_twotheta(axis.units):
        return r"Scattering Angle 2$\theta$ ($^{\circ}$)"
    else:
        return axis.units


def generate_legend(workspace_name, integrated_dim, integration_start, integration_end):
    mappings = {"DeltaE": "E", "MomentumTransfer": "|Q|", "Degrees": r"2$\theta$"}
    integrated_dim = (
        mappings[integrated_dim] if integrated_dim in mappings else integrated_dim
    )
    return (
        workspace_name
        + " "
        + "%.2f" % integration_start
        + "<"
        + integrated_dim
        + "<"
        + "%.2f" % integration_end
    )


def get_recoil_label(key) -> str:
    if key in recoil_labels:
        label = recoil_labels[key]
    else:
        label = "Relative mass " + str(key)
    return label
