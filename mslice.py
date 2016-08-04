from PyQt4.QtGui import QMainWindow,QApplication
from mslice_ui import Ui_MainWindow
from widgets.projection.powder.powder_ui import Ui_Form as powder_ui

class MsliceGui(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MsliceGui,self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    qapp = QApplication([])
    mslice = MsliceGui()
    mslice.show()
    qapp.exec_()