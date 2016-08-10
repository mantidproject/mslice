from workspace_provider import WorkspaceProvider
from mantid.simpleapi import mtd,Load,DeleteWorkspace,Load,GroupWorkspaces,RenameWorkspace,SaveNexus

class MantidWorkspaceProvider(WorkspaceProvider):

    def get_workspace_names(self):
        return mtd.getObjectNames()

    def delete_workspace(self, ToBeDeleted):
        return DeleteWorkspace(ToBeDeleted)

    def load(self, Filename, OutputWorkspace):
        return Load(Filename=Filename,OutputWorkspace=OutputWorkspace)

    def group_workspaces(self, InputWorkspaces, OutputWorkspace):
        return GroupWorkspaces(InputWorkspaces,OutputWorkspace)

    def rename_workspace(self, selected_workspace, newName):
        return RenameWorkspace(selected_workspace,newName)

    def save_nexus(self, workspace, path):
        SaveNexus(workspace,path)