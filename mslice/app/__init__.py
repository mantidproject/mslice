"""Package defining top-level MSlice application
and entry points.
"""

# Module-level reference to keep main window alive after show_gui has returned
MAIN_WINDOW = None
MPL_COMPAT = False

def check_mpl():
    from distutils.version import LooseVersion
    import matplotlib
    if LooseVersion(matplotlib.__version__) < LooseVersion("1.5.0"):
        import warnings
        warnings.warn('')
        warnings.warn('A version of Matplotlib older than 1.5.0 has been detected.', ImportWarning)
        warnings.warn('Some features of MSlice may not work correctly.', ImportWarning)
        warnings.warn('')
        global MPL_COMPAT
        MPL_COMPAT = True

def show_gui():
    """Display the top-level main window. It assumes that the QApplication instance
    has already been started
    """
    global MAIN_WINDOW
    if MAIN_WINDOW is None:
        from mslice.app.mainwindow import MainWindow
        MAIN_WINDOW = MainWindow()
    MAIN_WINDOW.show()

def startup(with_ipython):
    """Perform a full application startup, including the IPython
    shell if requested. If IPython is requested then the matplotlib
    backend is set to qt4. If IPython is not requested then the
    QApplication event loop is started manually
    :param with_ipython: If true then the IPython shell is started and
    mslice is launched from here
    """
    check_mpl()
    if with_ipython:
        import IPython
        IPython.start_ipython(["--matplotlib=qt4", "-i",
                               "-c from mslice.app import show_gui; show_gui()"])
    else:
        from PyQt4.QtGui import QApplication
        qapp = QApplication([])
        show_gui()
        qapp.exec_()
