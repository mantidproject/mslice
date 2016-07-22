from workspacemanager.WorkspaceView import WorkspaceView
from MainPresenter import MainPresenter
class MainView:
    def __init__(self):
        print 'mv created'
        self.x = 0
        print 'x= ',self.x,self._default_handler
        #TODO NO NO NO underscore is important
        self._workspace_manager_view = WorkspaceView()
        self._presenter = MainPresenter(self, self._workspace_manager_presenter)

    def get_presenter(self):
        return self._workspace_manager_view.get_presenter()
