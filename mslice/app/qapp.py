from mslice.util.qt.QtWidgets import QApplication

qApp = None


def create_qapp_if_required():
    """
    Create a qapplication for non gui plots
    """
    global qApp

    if qApp is None:
        instance = QApplication.instance()
        if instance is None:
            instance = QApplication(['mslice'])
            instance.lastWindowClosed.connect(instance.quit)
        qApp = instance
    return qApp


def show():
    if qApp is not None:
        return qApp.exec_()
