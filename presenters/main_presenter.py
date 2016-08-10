class MainPresenter():
    def __init__(self,main_view,workspace_presenter):
        self._mainView = main_view
        self._workspace_presenter = workspace_presenter

    def get_selected_workspaces(self):
        return self._workspace_presenter.get_selected_workspaces()

    def update_displayed_workspaces(self):
        """Update the workspaces shown to user.

        This function must be called by any presenter that
        does any operation that changes the name or type of any existing workspace or creates or removes a
        workspace"""
        self._workspace_presenter.update_displayed_workspaces()