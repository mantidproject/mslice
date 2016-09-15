import os
import_cli = True


if 'MSLICE_PLOT_WINDOWS' in os.environ.keys():
    if os.environ['MSLICE_PLOT_WINDOWS'] == 'TEST':
        import_cli = False

# The if is there to stop this script from running during tests. When this __init__ files runs it will import concrete
# Qt and Mantid implementations. This will fail on the build server

if import_cli:
    __all__ = []
    import plotting.pyplot as _plt
    from plotting.pyplot import *

    __all__ += dir(_plt)

    import _mslice_commmands
    from _mslice_commmands import *

    __all__ += dir(_mslice_commmands)