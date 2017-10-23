import abc
from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class MainPresenterInterface(object):

    @abc.abstractmethod
    def get_selected_workspaces(self):
        pass

    @abc.abstractmethod
    def set_selected_workspaces(self, workspace_list):
        pass

    @abc.abstractmethod
    def update_displayed_workspaces(self):
        pass

    @abc.abstractmethod
    def notify_workspace_selection_changed(self):
        pass

    @abc.abstractmethod
    def subscribe_to_workspace_selection_monitor(self, client):
        pass

    @abc.abstractmethod
    def register_workspace_selector(self, workspace_selector):
        pass
