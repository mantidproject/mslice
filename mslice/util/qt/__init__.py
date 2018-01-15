from .functions import load_ui
try:
    from qtpy import QT_VERSION as QT_VERSION
except (ImportError, ValueError):
    from pyqt4.Qt import QT_VERSION_STR as QT_VERSION
