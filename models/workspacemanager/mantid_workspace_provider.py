from workspace_provider import WorkspaceProvider
from mantid.simpleapi import AnalysisDataService,DeleteWorkspace,GroupWorkspaces,Load,RenameWorkspace,SaveNexus

class MantidWorkspaceProvider(WorkspaceProvider):

    def get_workspace_names(self):
        return AnalysisDataService.getObjectNames()

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


    def get_workspace_handle(self, workspace_name):
        """"Return handle to workspace given workspace_name_as_string"""
        return AnalysisDataService[workspace_name]