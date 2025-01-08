# system imports
import importlib

# local imports
from qtpy import QT_VERSION


def get_canvas_and_toolbar_cls():
    """
    Return the FigureCanvas and NavigationToolbar types appropriate for this instance
    :return: A 2-tuple of (FigureCanvas, NavigationToolbar)
    """
    backend = get_backend_module()
    return getattr(backend, 'FigureCanvas'), getattr(backend, 'NavigationToolbar2QT')


def get_backend_module():
    """
    Import the appropriate backend for the running version of Qt

    :return: A reference to the appropriate backend module
    """
    # Pick the relevant QtAgg one for the version we are running
    return importlib.import_module('matplotlib.backends.backend_qt{}agg'.format(QT_VERSION[0]))
