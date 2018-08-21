"""
Colour-map related functionality
"""
from mslice.util import MPL_COMPAT

ALLOWED_CMAPS = ['jet', 'summer', 'winter', 'coolwarm']
if not MPL_COMPAT:
    ALLOWED_CMAPS.insert(0, 'viridis')

DEFAULT_CMAP = ALLOWED_CMAPS[0]


def allowed_cmaps():
    """Return the allowed list of colour maps"""
    cmaps = ['viridis', 'jet', 'summer', 'winter', 'coolwarm']
    if MPL_COMPAT:
        cmaps.remove('viridis')
    return cmaps
