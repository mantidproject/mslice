from workspacemanager.WorkspaceManagerQuickView import WorkspaceManagerQuickView
from powderprojection.PowderProjectionQuickView import PowderProjectionQuickView
from MainPresenter import MainPresenter
import workspacemanager,powderprojection
from quickview.QuickView import QuickView
from mainview import MainView

#TODO Error in logging framework, FIX/ASK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class MainQuickView(MainView,QuickView):
    def __init__(self):
        #Might need to call QuickView.__init__ at some point, maybe to mock menubar calls?
        #Order OF Follwing statements is important!!!
        super(MainQuickView,self).__init__()
        print 'mv clobbered'
        self.workspace_manager_view = WorkspaceManagerQuickView(workspacemanager.command.Command)
        self._workspace_manager_presenter = self.workspace_manager_view.get_presenter()
        print id(self._workspace_manager_presenter)
        self.powder_projection_view = PowderProjectionQuickView(self, powderprojection.command.Command)
        self._presenter = MainPresenter(self, self._workspace_manager_presenter)
        # TODO add other subviews



if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    app = QApplication([])
    m = MainQuickView()
    print MainQuickView.__mro__
    print '***********************************\n'
    app.exec_()