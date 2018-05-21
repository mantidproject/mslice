"""Package defining top-level MSlice application
and entry points.
"""

from mslice.util.qt.QtWidgets import QApplication
from mslice.util.mantid.init_mantid import initialize_mantid

# Module-level reference to keep main window alive after show_gui has returned
MAIN_WINDOW = None

def main():
    """Start the application.
    """
    qapp_ref = QApplication([])
    show_gui()
    return qapp_ref.exec_()

def show_gui():
    """Display the top-level main window.
    If this is the first call then an instance of the Windows is cached to ensure it
    survives for the duration of the application
    """
    initialize_mantid()
    global MAIN_WINDOW
    if MAIN_WINDOW is None:
        from mslice.app.mainwindow import MainWindow
        MAIN_WINDOW = MainWindow()
    MAIN_WINDOW.show()
