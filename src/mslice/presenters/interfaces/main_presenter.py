from abc import ABCMeta, abstractmethod


class MainPresenterInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_selected_workspaces(self):
        pass

    @abstractmethod
    def set_selected_workspaces(self, workspace_list):
        pass

    @abstractmethod
    def update_displayed_workspaces(self):
        pass

    @abstractmethod
    def notify_workspace_selection_changed(self):
        pass

    @abstractmethod
    def subscribe_to_workspace_selection_monitor(self, client):
        pass

    @abstractmethod
    def register_workspace_selector(self, workspace_selector):
        pass

    @abstractmethod
    def change_ws_tab(self, tab):
        pass

    @abstractmethod
    def highlight_ws_tab(self, tab):
        pass

    @abstractmethod
    def show_workspace_manager_tab(self):
        pass

    @abstractmethod
    def show_tab_for_workspace(self, ws):
        pass

    @abstractmethod
    def subscribe_to_energy_default_monitor(self, client):
        pass

    @abstractmethod
    def subscribe_to_cut_algo_default_monitor(self, client):
        pass

    @abstractmethod
    def is_energy_conversion_allowed(self):
        pass

    @abstractmethod
    def get_cut_algorithm(self):
        pass
