try:
    from qtpy.uic import * # noqa: F401
except (ImportError, ValueError):
    from PyQt4.uic import * # noqa: F401
