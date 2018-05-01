"""Package defining top-level MSlice application
and entry points.
"""

from mantid.api import AlgorithmFactory
from mslice.util.qt.QtWidgets import QApplication
from mantid.simpleapi import _translate
from mslice.models.projection.powder.make_projection import MakeProjection
from mslice.models.slice.slice import Slice


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
    AlgorithmFactory.subscribe(MakeProjection)
    AlgorithmFactory.subscribe(Slice)
    _translate()
    global MAIN_WINDOW
    if MAIN_WINDOW is None:
        from mslice.app.mainwindow import MainWindow
        MAIN_WINDOW = MainWindow()
    MAIN_WINDOW.show()
