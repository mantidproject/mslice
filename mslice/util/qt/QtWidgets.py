try:
    from qtpy.QtWidgets import * # noqa: F401
    from qtpy.QtPrintSupport import * # noqa: F401
except (ImportError, ValueError):
    from PyQt4.QtGui import * # noqa: F401
