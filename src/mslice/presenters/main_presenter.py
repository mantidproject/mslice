from .interfaces.main_presenter import MainPresenterInterface
from mantid.api import IMDHistoWorkspace, IMDEventWorkspace
from collections.abc import Callable


class MainPresenter(MainPresenterInterface):
    def __init__(self, main_view, *subpresenters):
        self._mainView = main_view
        self._selected_workspace_listener = []
        self._energy_default_listener = []
        self._cut_algo_default_listener = []
        for presenter in subpresenters:
            presenter.register_master(self)

    def get_selected_workspaces(self):
        return self._workspace_presenter.get_selected_workspaces()

    def change_ws_tab(self, tab):
        self._workspace_presenter.change_tab(tab)

    def highlight_ws_tab(self, tab):
        self._workspace_presenter.highlight_tab(tab)

    def show_workspace_manager_tab(self):
        self._mainView.change_main_tab(1)

    def show_tab_for_workspace(self, ws):
        tab = 0
        if isinstance(ws, IMDHistoWorkspace):
            tab = 2
        elif isinstance(ws, IMDEventWorkspace):
            tab = 1
        self.change_ws_tab(tab)
        self._workspace_presenter.set_selected_workspaces([ws])

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
        """Subscribe a client to be notified when selected workspaces change
        client.workspace_selection_changed() will be called whenever the selected workspaces change"""
        if isinstance(getattr(client, "workspace_selection_changed", None), Callable):
            self._selected_workspace_listener.append(client)
        else:
            raise TypeError(
                "The client trying to subscribe does not implement the method 'workspace_selection_changed'"
            )

    def register_workspace_selector(self, workspace_selector):
        self._workspace_presenter = workspace_selector

    def subscribe_to_energy_default_monitor(self, client):
        if isinstance(getattr(client, "set_energy_default", None), Callable):
            self._energy_default_listener.append(client)

    def set_energy_default(self, en_default):
        for listener in self._energy_default_listener:
            listener.set_energy_default(en_default)

    def subscribe_to_cut_algo_default_monitor(self, client):
        if isinstance(getattr(client, "set_cut_algorithm_default", None), Callable):
            self._cut_algo_default_listener.append(client)

    def set_cut_algorithm_default(self, algo_default):
        for listener in self._cut_algo_default_listener:
            listener.set_cut_algorithm_default(algo_default)

    def is_energy_conversion_allowed(self):
        return self._mainView.is_energy_conversion_allowed()

    def get_cut_algorithm(self):
        return self._mainView.get_cut_algorithm()
