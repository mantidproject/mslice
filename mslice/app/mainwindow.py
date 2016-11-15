from PyQt4.QtGui import QMainWindow

from mslice.mainview import MainView
from mslice.presenters.main_presenter import MainPresenter
from .mainwindow_ui import Ui_MainWindow

# ==============================================================================
# Classes
# ==============================================================================

class MainWindow(QMainWindow, Ui_MainWindow, MainView):

    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)

        workspace_presenter = self.wgtWorkspacemanager.get_presenter()
        slice_presenter = self.wgtSlice.get_presenter()
        powder_presenter = self.wgtPowder.get_presenter()
        cut_presenter = self.wgtCut.get_presenter()
        self._presenter = MainPresenter(self, workspace_presenter, slice_presenter , powder_presenter, cut_presenter)

        self.wgtCut.error_occurred.connect(self.show_error)
        self.wgtSlice.error_occurred.connect(self.show_error)
        self.wgtWorkspacemanager.error_occurred.connect(self.show_error)
        self.wgtPowder.error_occurred.connect(self.show_error)

    def show_error(self, msg):
        """Show an error message on status bar. If msg ==""  the function will clear the displayed message """
        self.statusbar.showMessage(msg)

    def get_presenter(self):
        return self._presenter
