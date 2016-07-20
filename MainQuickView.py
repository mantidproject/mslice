from workspacemanager.WorkspaceManagerQuickView import WorkspaceManagerQuickView
from powderprojection.PowderProjectionQuickView import PowderProjectionQuickView
from MainPresenter import MainPresenter
import workspacemanager,powderprojection

#TODO Error in logging framework, FIX/ASK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class MainQuickView():
    def __init__(self):

        self._presenter = MainPresenter(self)
        self.WorkspaceManagerView = WorkspaceManagerQuickView(workspacemanager.command.Command)
        self._workspaceManagerPresenter = self.WorkspaceManagerView.get_presenter()
        self.PowderProjectionView = PowderProjectionQuickView(powderprojection.command.Command)
        # TODO add other subviews


    def get_workspace_manager_presenter(self):
        return self._workspaceManagerPresenter

if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    m = MainQuickView()
    app.exec_()