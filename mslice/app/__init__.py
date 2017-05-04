"""Package defining top-level MSlice application
and entry points.
"""

# Module-level reference to keep main window alive after show_gui has returned
MAIN_WINDOW = None

def show_gui():
    """Display the top-level main window. If the Qt event loop has not been started, starts it.
    """
    from PyQt4.QtGui import QApplication
    qapp = QApplication.instance()
    runningIP = False
    runExec = False
    if not qapp:
        # Check that if we're running in IPython, try to use IPython's event loop
        try:
            ip = get_ipython()
            if qapp is None:
                ip.enable_matplotlib('qt4')
            runningIP = True
        except NameError:
            pass
    # Also catch case where IPython event loop fails
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication([])
        runExec = True

    global MAIN_WINDOW
    if MAIN_WINDOW is None:
        from mslice.app.mainwindow import MainWindow
        MAIN_WINDOW = MainWindow()
    MAIN_WINDOW.show()

    if runExec:
        if runningIP:
            import warnings
            warnings.warn('Unable to start IPython GUI event loop. Scripting not available')
        qapp.exec_()


def startup(with_ipython):
    """Perform a full application startup, including the IPython
    shell if requested. If IPython is requested then the matplotlib
    backend is set to qt4. If IPython is not requested then the
    QApplication event loop is started manually
    :param with_ipython: If true then the IPython shell is started and
    mslice is launched from here
    """
    # Checks if running within MantidPlot, if so, just run show_gui()
    try:
        import mantidplot # noqa
        show_gui()
    except:
        if with_ipython:
            # Check that we are not already running IPython!
            try:
                get_ipython()
                show_gui()
            except NameError:
                import IPython
                IPython.start_ipython(["--matplotlib=qt4", "-i",
                                       "-c from mslice.app import show_gui; show_gui()"])
        else:
            from PyQt4.QtGui import QApplication
            if QApplication.instance():
                qapp = QApplication.instance()
            else:
                qapp = QApplication([])
            show_gui()
            qapp.exec_()
