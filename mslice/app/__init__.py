"""Package defining top-level MSlice application
and entry points.
"""
import os

from qtpy.QtWidgets import QApplication

from mslice.external.ipython import start_ipython

# Module-level reference to keep main window alive after show_gui has returned
MAIN_WINDOW = None
QAPP_REF = None
MPL_COMPAT = False

def check_mpl():
    from distutils.version import LooseVersion
    import matplotlib
    if LooseVersion(matplotlib.__version__) < LooseVersion("1.5.0"):
        import warnings
        warnings.warn('A version of Matplotlib older than 1.5.0 has been detected.')
        warnings.warn('Some features of MSlice may not work correctly.')
        global MPL_COMPAT
        MPL_COMPAT = True

def main():
    """Start the application. Detects the current environment and
    runs accordingly:
      - if an existing QApplication is detected then this is used and IPython
      - is not started, otherwise  a application is created.
      - if an existing IPython shell is detected this instance is used
        and matplotlib support is enabled otherwise a new one is created
    """
    check_mpl()
    global QAPP_REF
    # if QApplication.instance():
        # We must be embedded in some other application that has already started the event loop
        # just show the UI...
    QAPP_REF = QApplication([])
    show_gui()
    return QAPP_REF.exec_()

def show_gui():
    """Display the top-level main window.
    If this is the first call then an instance of the Windows is cached to ensure it
    survives for the duration of the application
    """
    global MAIN_WINDOW
    if MAIN_WINDOW is None:
        from mslice.app.mainwindow import MainWindow
        MAIN_WINDOW = MainWindow()
    MAIN_WINDOW.show()
