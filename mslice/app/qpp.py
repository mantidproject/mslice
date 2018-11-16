from mslice.util.qt.QtWidgets import QApplication


qApp = None

CLI_MAIN = None


def create_qapp():
    """
    Create a qapplication
    """
    global qApp

    if qApp is None:
        qApp = QApplication([])
        add_main()
    #return qApp.exec_()


def add_main():
    pass


def close():
    global qApp
    qApp.quit()
