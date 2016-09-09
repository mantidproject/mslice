import os
import_plotting_functions = True

#importing plotting.pyplot will import qt. This will fail on CI servers if pyqt is unavailable
if 'MSLICE_PLOT_WINDOWS' in os.environ.keys():
    if os.environ['MSLICE_PLOT_WINDOWS'] == 'TEST':
        import_plotting_functions = False

__all__ = []
if import_plotting_functions:
    import plotting.pyplot as _plt
    from plotting.pyplot import *

    __all__ += dir(_plt)

import _mslice_commmands
from _mslice_commmands import *

__all__ += dir(_mslice_commmands)