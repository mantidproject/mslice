try:
    from qtpy.QtGui import * # noqa: F401
except (ImportError, ValueError):
    from PyQt4.QtGui import * # noqa: F401
