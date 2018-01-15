try:
    from qtpy.QtCore import *
except (ImportError, ValueError):
    from PyQt4.QtCore import *
    from PyQt4.QtCore import pyqtSignal as Signal
