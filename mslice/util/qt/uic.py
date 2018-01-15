try:
    from qtpy.uic import *
except (ImportError, ValueError):
    from PyQt4.uic import *
