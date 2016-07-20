from WorkspaceManager.WorkspaceManagerQuickView import WorkspaceManagerQuickView
from PowderProjection.PowderProjectionQuickView import PowderProjectionQuickView
from MainPresenter import MainPresenter
import WorkspaceManager,PowderProjection

#TODO Error in logging framework !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class MainQuickView():
    def __init__(self):

        self._presenter = MainPresenter(self)
        self.WorkspaceManagerView = WorkspaceManagerQuickView(WorkspaceManager.command.Command)
        self._workspaceManagerPresenter = self.WorkspaceManagerView.get_presenter()
        self.PowderProjectionView = PowderProjectionQuickView(PowderProjection.command.Command)
        # TODO add other subviews


    def get_workspace_manager_presenter(self):
        return self._workspaceManagerPresenter

if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    m = MainQuickView()
    app.exec_()