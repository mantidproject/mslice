from .interfaces.main_presenter import MainPresenterInterface


class MainPresenter(MainPresenterInterface):
    def __init__(self, main_view, *subpresenters):
        self._mainView = main_view
        self._selected_workspace_listener = []
        for presenter in subpresenters:
            presenter.register_master(self)

    def get_selected_workspaces(self):
        return self._workspace_presenter.get_selected_workspaces()

    def set_selected_workspaces(self, workspace_list):
        self._workspace_presenter.set_selected_workspaces(workspace_list)

    def update_displayed_workspaces(self):
        """Update the workspaces shown to user.

        This function must be called by any presenter that
        does any operation that changes the name or type of any existing workspace or creates or removes a
        workspace"""
        self._workspace_presenter.update_displayed_workspaces()

    def broadcast_selection_changed(self):
        for listener in self._selected_workspace_listener:
            listener.workspace_selection_changed()

    def notify_workspace_selection_changed(self):
        self.broadcast_selection_changed()

    def subscribe_to_workspace_selection_monitor(self, client):
        """Subcscribe a client to be notified when selected workspaces change

        client.workspace_selection_changed() will be called whenever the selected workspaces change"""
        if callable(getattr(client, "workspace_selection_changed",None)):
            self._selected_workspace_listener.append(client)
        else:
            raise TypeError("The client trying to subscribe does not implement the method 'workspace_selection_changed'")

    def register_workspace_selector(self, workspace_selector):
        self._workspace_presenter = workspace_selector
