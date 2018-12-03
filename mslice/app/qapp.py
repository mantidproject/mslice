from mslice.util.qt.QtWidgets import QApplication
from mslice.app.presenters import is_gui

qApp = None


def create_qapp_if_required():
    """
    Create a qapplication for non gui plots
    """
    global qApp
    if is_gui():
        return
    if qApp is None:
        qApp = QApplication([])
        qApp.lastWindowClosed.connect(qApp.quit)
    return qApp


def show():
    if qApp is not None:
        return qApp.exec_()
