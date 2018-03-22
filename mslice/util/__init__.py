from distutils.version import LooseVersion
import matplotlib

MPL_COMPAT = False

# check matplotlib version
if LooseVersion(matplotlib.__version__) < LooseVersion("1.5.0"):
    import warnings
    warnings.warn('A version of Matplotlib older than 1.5.0 has been detected.')
    warnings.warn('Some features of MSlice may not work correctly.')
    MPL_COMPAT = True
