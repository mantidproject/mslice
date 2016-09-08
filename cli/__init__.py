import os
if os.environ['MSLICE_PLOT_WINDOWS'] != 'TEST':
    import plotting.pyplot as _plt
    from plotting.pyplot import *

    __all__ = dir(_plt)