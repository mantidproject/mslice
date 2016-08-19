from PyQt4.QtGui import QMainWindow,QApplication

from mainview import MainView
from mslice_ui import Ui_MainWindow
from presenters.main_presenter import MainPresenter


class MsliceGui(QMainWindow,Ui_MainWindow,MainView):
    def __init__(self):
        super(MsliceGui,self).__init__()
        self.setupUi(self)
        self._presenter = MainPresenter(self)

        self.wgtWorkspacemanager.get_presenter().register_master(self)
        self.wgtSlice.get_presenter().register_master(self)
        self.wgtPowder.get_presenter().register_master(self)

    def get_presenter(self):
        return self._presenter

if __name__ == "__main__":
    qapp = QApplication([])
    mslice = MsliceGui()
    mslice.show()
    qapp.exec_()