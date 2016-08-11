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
        self.wgtSlice.set_main_window(self)
        self.wgtPowder.set_main_window(self)

if __name__ == "__main__":
    from build_all_ui_files import build_all_ui_files,script_folder
    build_all_ui_files(script_folder)
    qapp = QApplication([])
    mslice = MsliceGui()
    mslice.show()
    qapp.exec_()