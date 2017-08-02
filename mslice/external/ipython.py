"""This module contains patches to IPython to avoid bugs with various
versions. Client code shold import from this module
"""
# std library
from distutils.version import LooseVersion

# third party
import IPython
# define same public api as IPython
from IPython import * # noqa

__all__ = dir(IPython)

if LooseVersion(IPython.__version__) >= LooseVersion("1.0.0") and \
   LooseVersion(IPython.__version__) < LooseVersion("3.0.0"):
    # Monkey patch IPython.external.qt_loaders.commit_api to avoid
    # a bugs on IPython various 1 < v < 3
    # See https://github.com/ipython/ipython/pull/6730
    import IPython.external.qt_loaders as _qt_loaders
    def _commit_api_patched(api):
        """Commit to a particular API, and trigger ImportErrors on subsequent
           dangerous imports"""
        ID =  _qt_loaders.ID
        if api == _qt_loaders.QT_API_PYSIDE:
            ID.forbid('PyQt4')
            ID.forbid('PyQt5')
        elif api == _qt_loaders.QT_API_PYQT5:
            ID.forbid('PySide')
            ID.forbid('PyQt4')
        else:   # There are three other possibilities, all representing PyQt4
            ID.forbid('PyQt5')
            ID.forbid('PySide')
    # enddef
    # patch
    _qt_loaders.commit_api = _commit_api_patched
