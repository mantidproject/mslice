from workspacemanager.command import Command
import os.path


#TODO askOwen is passing mtd.getObjectNames() ideal? pass mtd? Answer: Will do for time being
#TODO When moving to actual workspaces the tests should reflect that
#TODO implement compose workspace
#TODO tell user when file not found,file failed to load

class WorkspaceManagerPresenter(object):
    def __init__(self,view,workspace_provider=None):
        self._groupCount = 1
        self._workspaceMangerView = view
        self._workSpaceprovider = workspace_provider

    def notify(self,command):
        if command == Command.LoadWorkspace:
            #TODO specify workspace name on load
            #TODO what to do on fail?
            workspace_to_load = self._workspaceMangerView.get_workspace_to_load_path()
            base = os.path.basename(workspace_to_load)
            ws_name = os.path.splitext(base)[0]
            self._workSpaceprovider.Load(Filename=workspace_to_load, OutputWorkspace=ws_name)
            self._workspaceMangerView.display_loaded_workspaces(self._workSpaceprovider.getWorkspaceNames())

        if command == Command.SaveSelectedWorkspace:
            selected_workspaces = self._workspaceMangerView.get_workspace_selected()
            if not selected_workspaces:
                self._workspaceMangerView.error_select_one_workspace()
                return
            if len(selected_workspaces)>1:
                self._workspaceMangerView.error_select_only_one_workspace()
                return
            selected_workspace = selected_workspaces[0]
            path = self._workspaceMangerView.get_workspace_to_save_filepath()
            self._workSpaceprovider.SaveNexus(selected_workspace, path)

        if command == Command.RemoveSelectedWorkspaces:
            selected_workspaces = self._workspaceMangerView.get_workspace_selected()
            if not selected_workspaces:
                self._workspaceMangerView.error_select_one_or_more_workspaces()
                return
            for workspace in selected_workspaces:
                self._workSpaceprovider.DeleteWorkspace(workspace)
            self._workspaceMangerView.display_loaded_workspaces(self._workSpaceprovider.getWorkspaceNames())

        if command == Command.GroupSelectedWorkSpaces:
            selected_workspaces = self._workspaceMangerView.get_workspace_selected()
            if not selected_workspaces:
                self._workspaceMangerView.error_select_one_or_more_workspaces()
                return
            groupName = 'group' + str(self._groupCount)
            self._groupCount+=1
            self._workSpaceprovider.GroupWorkspaces(InputWorkspaces = selected_workspaces, OutputWorkspace = groupName)
            self._workspaceMangerView.display_loaded_workspaces(self._workSpaceprovider.getWorkspaceNames())

        if command == Command.RenameWorkspace:
            selected_workspaces = self._workspaceMangerView.get_workspace_selected()
            if not selected_workspaces:
                self._workspaceMangerView.error_select_one_workspace()
                return
            if len(selected_workspaces)>1:
                self._workspaceMangerView.error_select_only_one_workspace()
                return
            selected_workspace = selected_workspaces[0]
            newName = self._workspaceMangerView.get_workspace_new_name()
            self._workSpaceprovider.RenameWorkspace(selected_workspace, newName)
            self._workspaceMangerView.display_loaded_workspaces(self._workSpaceprovider.getWorkspaceNames())


