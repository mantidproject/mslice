from mslice.util.qt.QtWidgets import QApplication

qApp = None


def create_qapp():
    """
    Create a qapplication for non gui plots
    """
    global qApp

    if qApp is None:
        qApp = QApplication([])
        qApp.lastWindowClosed.connect(qApp.quit)
    return qApp


def show():
    if qApp is not None:
        return qApp.exec_()
