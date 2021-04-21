"""Package defining top-level MSlice application
and entry points.
"""
import sys

import mslice.util.mantid.init_mantid # noqa: F401
from mslice.util.mantid import in_mantid
from mslice.util.qt import QT_VERSION
from mslice.util.qt.qapp import create_qapp_if_required


# Module-level reference to keep main window alive after show_gui has returned
MAIN_WINDOW = None


def is_gui():
    """Check if mainwindow is instantiated"""
    return MAIN_WINDOW is not None


def main():
    """Start the application.
    """
    qapp_ref = create_qapp_if_required()
    import matplotlib as mpl
    mpl.use('Qt{}Agg'.format(QT_VERSION[0]))
    show_gui()
    return qapp_ref.exec_()


def show_gui():
    """Display the top-level main window.
    If this is the first call then an instance of the Windows is cached to ensure it
    survives for the duration of the application
    """
    global MAIN_WINDOW
    if MAIN_WINDOW is None:
        from mslice.app.mainwindow import MainWindow
        MAIN_WINDOW = MainWindow(in_mantid())

    if 'workbench' in sys.modules:
        from workbench.config import get_window_config

        # Ensures any change to workbench Floating/Ontop window setting is propagated to MSlice.
        parent, flags = get_window_config()
        MAIN_WINDOW.setParent(parent)
        MAIN_WINDOW.setWindowFlags(flags)
    MAIN_WINDOW.show()
