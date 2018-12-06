"""Package defining top-level MSlice application
and entry points.
"""
import mslice.util.mantid.init_mantid # noqa: F401
from mslice.util.mantid import in_mantidplot
from mslice.app.qapp import create_qapp_if_required

# Module-level reference to keep main window alive after show_gui has returned
MAIN_WINDOW = None


def main():
    """Start the application.
    """
    qapp_ref = create_qapp_if_required()
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
        MAIN_WINDOW = MainWindow(in_mantidplot())
    MAIN_WINDOW.show()
