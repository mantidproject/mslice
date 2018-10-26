"""Module to create axis and legend labels"""

from __future__ import (absolute_import, division, print_function)
from mslice.util import MPL_COMPAT

CUT_INTENSITY_LABEL = 'Signal/#Events'
recoil_labels = {1: 'Hydrogen', 2: 'Deuterium', 4: 'Helium'}

# unit labels for projections
MOD_Q_LABEL = '|Q|'
THETA_LABEL = '2Theta'
DELTA_E_LABEL = 'DeltaE'
MEV_LABEL = 'meV'
WAVENUMBER_LABEL = 'cm-1'


def get_display_name(axisUnits, comment=None):
    if 'DeltaE' in axisUnits:
        # Matplotlib 1.3 doesn't handle LaTeX very well. Sometimes no legend appears if we use LaTeX
        if MPL_COMPAT:
            return 'Energy Transfer ' + ('(cm-1)' if (comment and 'wavenumber' in comment) else '(meV)')
        else:
            return 'Energy Transfer ' + ('(cm$^{-1}$)' if (comment and 'wavenumber' in comment) else '(meV)')
    elif 'MomentumTransfer' in axisUnits or '|Q|' in axisUnits:
        return '|Q| (recip. Ang.)' if MPL_COMPAT else r'$|Q|$ ($\mathrm{\AA}^{-1}$)'
    elif '2Theta' in axisUnits:
        return 'Scattering Angle (degrees)' if MPL_COMPAT else r'Scattering Angle 2$\theta$ ($^{\circ}$)'
    else:
        return axisUnits


def generate_legend(workspace_name, integrated_dim, integration_start, integration_end):
    if MPL_COMPAT:
        mappings = {'DeltaE': 'E', 'MomentumTransfer': '|Q|', 'Degrees': r'2Theta'}
    else:
        mappings = {'DeltaE': 'E', 'MomentumTransfer': '|Q|', 'Degrees': r'2$\theta$'}
    integrated_dim = mappings[integrated_dim] if integrated_dim in mappings else integrated_dim
    return workspace_name + " " + "%.2f" % integration_start + "<" + integrated_dim + "<" + \
        "%.2f" % integration_end
