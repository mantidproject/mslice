from workspacemanager.WorkspaceView import WorkspaceView
from MainPresenter import MainPresenter


class MainView:
    def __init__(self):
        self._workspace_manager_view = WorkspaceView()
        workspace_manager_presenter = self._workspace_manager_view.get_presenter()
        self._presenter = MainPresenter(self, workspace_manager_presenter)

    def get_presenter(self):
        return self._presenter
