"""Module to create axis and legend labels"""

from __future__ import (absolute_import, division, print_function)
from mslice.util import MPL_COMPAT
from mslice.models.units import EnergyUnits

CUT_INTENSITY_LABEL = 'Signal/#Events'
recoil_labels = {1: 'Hydrogen', 2: 'Deuterium', 4: 'Helium'}

# unit labels for projections
MOD_Q_LABEL = '|Q|'
THETA_LABEL = '2Theta'
DELTA_E_LABEL = 'DeltaE'


def get_recoil_key(label):
    for key, value in recoil_labels.items():
        if value == label:
            return key
    return label


def get_display_name(axis):
    if 'DeltaE' in axis.units:
        return EnergyUnits(axis.e_unit).label()
    elif 'MomentumTransfer' in axis.units or '|Q|' in axis.units:
        # Matplotlib 1.3 doesn't handle LaTeX very well. Sometimes no legend appears if we use LaTeX
        return '|Q| (recip. Ang.)' if MPL_COMPAT else r'$|Q|$ ($\mathrm{\AA}^{-1}$)'
    elif 'Degrees' in axis.units or '2Theta' in axis.units:
        return 'Scattering Angle (degrees)' if MPL_COMPAT else r'Scattering Angle 2$\theta$ ($^{\circ}$)'
    else:
        return axis.units


def generate_legend(workspace_name, integrated_dim, integration_start, integration_end):
    if MPL_COMPAT:
        mappings = {'DeltaE': 'E', 'MomentumTransfer': '|Q|', 'Degrees': r'2Theta'}
    else:
        mappings = {'DeltaE': 'E', 'MomentumTransfer': '|Q|', 'Degrees': r'2$\theta$'}
    integrated_dim = mappings[integrated_dim] if integrated_dim in mappings else integrated_dim
    return workspace_name + " " + "%.2f" % integration_start + "<" + integrated_dim + "<" + \
        "%.2f" % integration_end
