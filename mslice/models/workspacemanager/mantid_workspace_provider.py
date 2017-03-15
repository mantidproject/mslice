"""A concrete implementation of a WorkspaceProvider

It uses mantid to perform the workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from mantid.simpleapi import (AnalysisDataService, DeleteWorkspace, Load,
                              RenameWorkspace, SaveNexus, SaveMD)
from mantid.api import IMDWorkspace, Workspace

from .workspace_provider import WorkspaceProvider

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class MantidWorkspaceProvider(WorkspaceProvider):
    def __init__(self):
        # Stores various parameters of workspaces not stored by Mantid
        self._EfDefined = {}

    def get_workspace_names(self):
        return AnalysisDataService.getObjectNames()

    def delete_workspace(self, workspace):
        return DeleteWorkspace(Workspace=workspace)

    def _processEfixed(self, workspace):
        """Checks whether the fixed energy is defined for this workspace"""
        ws_name = workspace if isinstance(workspace, basestring) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(ws_name)
        try:
            [ws_h.getEFixed(ws_h.getDetector(i).getID()) for i in range(ws_h.getNumberHistograms())]
            self._EfDefined[ws_name] = True
        except RuntimeError:
            self._EfDefined[ws_name] = False

    def load(self, filename, output_workspace):
        ws = Load(Filename=filename, OutputWorkspace=output_workspace)
        if self.get_emode(output_workspace) == 'Indirect':
            self._processEfixed(output_workspace)
        return ws

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

    def get_workspace_name(self, workspace):
        """Returns the name of a workspace given the workspace handle"""
        if isinstance(workspace, basestring):
            return workspace
        return workspace.name()

    def get_emode(self, workspace):
        """Returns the energy analysis mode (direct or indirect of a workspace)"""
        if isinstance(workspace, basestring):
            workspace_handle = self.get_workspace_handle(workspace)
        else:
            workspace_handle = workspace
        return workspace_handle.getEMode().name

    def has_efixed(self, workspace):
        return self._EfDefined[workspace if isinstance(workspace, basestring) else self.get_workspace_name(workspace)]

    def set_efixed(self, workspace, Ef):
        """Sets (overides) the fixed energy for all detectors (spectra) of this workspace"""
        ws_name = workspace if isinstance(workspace, basestring) else self.get_workspace_name(workspace)
        ws_handle = self.get_workspace_handle(ws_name)
        for idx in range(ws_handle.getNumberHistograms()):
            ws_handle.setEFixed(ws_handle.getDetector(idx).getID(), Ef)
