import abc

class WorkspaceProvider:
    @abc.abstractmethod
    def get_workspace_names(self):
        pass
    @abc.abstractmethod
    def delete_workspace(self, ToBeDeleted):
        pass
    @abc.abstractmethod
    def load(self, Filename, OutputWorkspace):
        pass
    @abc.abstractmethod
    def group_workspaces(self, InputWorkspaces, OutputWorkspace):
        pass
    @abc.abstractmethod
    def rename_workspace(self, selected_workspace, newName):
        pass
    @abc.abstractmethod
    def save_nexus(self, workspace, path):
        pass