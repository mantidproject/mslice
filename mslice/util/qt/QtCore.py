try:
    from qtpy.QtCore import * # noqa: F401
except (ImportError, ValueError):
    from PyQt4.QtCore import * # noqa: F401
    from PyQt4.QtCore import pyqtSignal as Signal # noqa: F401
