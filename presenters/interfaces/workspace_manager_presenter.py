class WorkspaceManagerPresenterInterface(object):
    def __init__(self, workspace_view, workspace_provider):
        raise Exception("Interface Class must not be instantiated")

    def register_master(self, main_view):
        raise NotImplementedError("This method must be overriden in implementation")

    def notify(self, command):
        raise NotImplementedError("This method must be overriden in implementation")

    def get_selected_workspaces(self):
        raise NotImplementedError("This method must be overriden in implementation")

    def set_selected_workspaces(self, list):
        raise NotImplementedError("This method must be overriden in implementation")

    def update_displayed_workspaces(self):
        raise NotImplementedError("This method must be overriden in implementation")
