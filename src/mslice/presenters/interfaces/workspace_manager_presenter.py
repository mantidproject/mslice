from abc import ABCMeta, abstractmethod


class WorkspaceManagerPresenterInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def register_master(self, main_view):
        pass

    @abstractmethod
    def notify(self, command):
        pass

    @abstractmethod
    def get_selected_workspaces(self):
        pass

    @abstractmethod
    def set_selected_workspaces(self, workspace_list):
        pass

    @abstractmethod
    def update_displayed_workspaces(self):
        pass
