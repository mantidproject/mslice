import abc
from six import add_metaclass

@add_metaclass(abc.ABCMeta)
class WorkspaceManagerPresenterInterface(object):

    @abc.abstractmethod
    def register_master(self, main_view):
        pass

    @abc.abstractmethod
    def notify(self, command):
        pass

    @abc.abstractmethod
    def get_selected_workspaces(self):
        pass

    @abc.abstractmethod
    def set_selected_workspaces(self, workspace_list):
        pass

    @abc.abstractmethod
    def get_workspace_provider(self):
        pass

    @abc.abstractmethod
    def update_displayed_workspaces(self):
        pass
