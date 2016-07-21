
class MainPresenter():
    def __init__(self,MainView,workspace_presenter):
        self._mainView = MainView
        self._workspace_presenter = workspace_presenter

    def get_selected_workspaces(self):
        return self._workspace_presenter.get_selected_workspaces()

    def refresh(self):
        self._workspace_presenter.refresh()