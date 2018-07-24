"""
Colour-map related functionality
"""
from mslice.util import MPL_COMPAT


def allowed_cmaps():
    """Return the allowed list of colour maps"""
    cmaps = ['viridis', 'jet', 'summer', 'winter', 'coolwarm']
    if MPL_COMPAT:
        cmaps.remove('viridis')
    return cmaps
