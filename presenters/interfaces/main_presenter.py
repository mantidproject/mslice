class MainPresenterInterface(object):
    def __init__(self, main_view):
        raise Exception("This class is an abstract interface and must no be instantiated")

    def get_selected_workspaces(self):
        raise NotImplementedError("This method must be overriden in implementation")

    def set_selected_workspaces(self, list):
        raise NotImplementedError("This method must be overriden in implementation")

    def update_displayed_workspaces(self):
        raise NotImplementedError("This method must be overriden in implementation")

    def notify_workspace_selection_changed(self):
        raise NotImplementedError("This method must be overriden in implementation")

    def subscribe_to_workspace_selection_monitor(self, client):
        raise NotImplementedError("This method must be overriden in implementation")

    def register_workspace_selector(self, workspace_selector):
        raise NotImplementedError("This method must be overriden in implementation")
