from workspacemanager.WorkspaceManagerQuickView import WorkspaceManagerQuickView
from powderprojection.PowderProjectionQuickView import PowderProjectionQuickView
from MainPresenter import MainPresenter
import workspacemanager,powderprojection
from quickview.QuickView import QuickView

#TODO Error in logging framework, FIX/ASK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class MainQuickView(QuickView):
    def __init__(self):
        #Might need to call super at some point, maybe to mock menubar calls?
        #Order OF Follwing statements is important!!! #TODO
        self.WorkspaceManagerView = WorkspaceManagerQuickView(workspacemanager.command.Command)
        self._workspaceManagerPresenter = self.WorkspaceManagerView.get_presenter()
        self.PowderProjectionView = PowderProjectionQuickView(self,powderprojection.command.Command)
        self._presenter = MainPresenter(self,self._workspaceManagerPresenter)
        # TODO add other subviews

    def get_workspace_manager_presenter(self):
        return self._workspaceManagerPresenter

    def get_presenter(self):
        return self._presenter

if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    m = MainQuickView()
    app.exec_()