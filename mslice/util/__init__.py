MPL_COMPAT = False

# check matplotlib version
from distutils.version import LooseVersion
import matplotlib
if LooseVersion(matplotlib.__version__) < LooseVersion("1.5.0"):
    import warnings
    warnings.warn('A version of Matplotlib older than 1.5.0 has been detected.')
    warnings.warn('Some features of MSlice may not work correctly.')
    global MPL_COMPAT
    MPL_COMPAT = True