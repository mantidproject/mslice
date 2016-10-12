"""A concrete implementation of a WorkspaceProvider

It uses mantid to perform the workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from mantid.simpleapi import (AnalysisDataService, DeleteWorkspace, Load,
                              RenameWorkspace, SaveNexus, SaveMD)
from mantid.api import IMDWorkspace, Workspace

from workspace_provider import WorkspaceProvider

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class MantidWorkspaceProvider(WorkspaceProvider):
    def get_workspace_names(self):
        return AnalysisDataService.getObjectNames()

    def delete_workspace(self, workspace):
        return DeleteWorkspace(Workspace=workspace)

    def load(self, filename, output_workspace):
        return Load(Filename=filename, OutputWorkspace=output_workspace)

    def rename_workspace(self, selected_workspace, new_name):
        return RenameWorkspace(InputWorkspace=selected_workspace, OutputWorkspace=new_name)

    def save_nexus(self, workspace, path):
        workspace_handle = self.get_workspace_handle(workspace)
        if isinstance(workspace_handle, IMDWorkspace):
            SaveMD(InputWorkspace=workspace, Filename=path)
        else:
            SaveNexus(InputWorkspace=workspace, Filename=path)

    def get_workspace_handle(self, workspace_name):
        """"Return handle to workspace given workspace_name_as_string"""

        # if passed a workspace handle return the handle
        if isinstance(workspace_name, Workspace):
            return workspace_name
        return AnalysisDataService[workspace_name]
