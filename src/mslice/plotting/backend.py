# system imports
import importlib


def get_canvas_and_toolbar_cls():
    """
    Return the FigureCanvas and NavigationToolbar types appropriate for this instance
    :return: A 2-tuple of (FigureCanvas, NavigationToolbar)
    """
    backend = get_backend_module()
    return backend.FigureCanvas, backend.NavigationToolbar2QT


def get_backend_module():
    """
    Import the appropriate backend for the running version of Qt

    :return: A reference to the appropriate backend module
    """
    # Pick the relevant QtAgg one for the version we are running
    return importlib.import_module("matplotlib.backends.backend_qtagg")
