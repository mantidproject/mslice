from .functions import load_ui # noqa: F401
try:
    from qtpy import QT_VERSION as QT_VERSION # noqa: F401
except (ImportError, ValueError):
    from PyQt4.Qt import QT_VERSION_STR as QT_VERSION # noqa: F401
