from PyQt4.QtGui import QMainWindow,QApplication

from mainview import MainView
from mslice_ui import Ui_MainWindow
from presenters.main_presenter import MainPresenter


class MsliceGui(QMainWindow,Ui_MainWindow,MainView):

    def __init__(self):
        super(MsliceGui,self).__init__()
        self.setupUi(self)
        workspace_presenter = self.wgtWorkspacemanager.get_presenter()
        self._presenter = MainPresenter(self,workspace_presenter)

        self.wgtWorkspacemanager.set_main_window(self)
        self.wgtSlice.set_workspace_selector(self)
        self.wgtSlice.error_occurred.connect(self.show_error)

    def show_error(self, error):
        self.statusbar.showMessage(error, 2000)


if __name__ == "__main__":
    qapp = QApplication([])
    mslice = MsliceGui()
    mslice.show()
    qapp.exec_()