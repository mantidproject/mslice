"""Package defining top-level MSlice application
and entry points.
"""

def show_gui():
    """Display the top-level main window. It assumes that the QApplication instance
    has already been started
    """
    from mslice.app.mainwindow import MainWindow
    win = MainWindow()
    win.show()

def startup(with_ipython):
    """Perform a full application startup, including the IPython
    shell if requested. If IPython is requested then the matplotlib
    backend is set to qt4. If IPython is not requested then the
    QApplication event loop is started manually
    :param with_ipython: If true then the IPython shell is started and
    mslice is launched from here
    """
    if with_ipython:
        import IPython
        IPython.start_ipython(["--matplotlib=qt4", "-i",
                               "-c from mslice.app import show_gui; show_gui()"])
    else:
        from PyQt4.QtGui import QApplication
        qapp = QApplication([])
        show_gui()
        qapp.exec_()
