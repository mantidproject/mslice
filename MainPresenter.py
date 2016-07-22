class MainPresenter():
    def __init__(self,main_view,workspace_presenter):
        self._mainView = main_view
        self._workspace_presenter = workspace_presenter

    def get_selected_workspaces(self):
        print 'I am going to call :', self._workspace_presenter._get_selected_workspaces
        return self._workspace_presenter._get_selected_workspaces()

    def refresh(self):
        self._workspace_presenter.update_displayed_workspaces()