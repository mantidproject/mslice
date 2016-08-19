from widgets.workspacemanager.command import Command
import os.path
from validation_decorators import require_main_presenter

#TODO tell user when file not found,file failed to load

class WorkspaceManagerPresenter(object):
    def __init__(self, workspace_view, workspace_provider):
        #TODO add validation checks
        self._groupCount = 1
        self._workspace_manger_view = workspace_view
        self._work_spaceprovider = workspace_provider
        self._main_presenter = None

    def register_master(self, main_view):
        self._main_presenter = main_view.get_presenter()
        self._main_presenter.register_workspace_selector(self)

    def notify(self, command):
        if command == Command.LoadWorkspace:
            self._load_workspace()
        elif command == Command.SaveSelectedWorkspace:
            self._save_selected_workspace()
        elif command == Command.RemoveSelectedWorkspaces:
            self._remove_selected_workspaces()
        elif command == Command.GroupSelectedWorkSpaces:
            self._group_selected_workspaces()
        elif command == Command.RenameWorkspace:
            self._rename_workspace()
        elif command == Command.SelectionChanged:
            self._broadcast_selected_workspaces()
        else:
            raise ValueError("Workspace Manager Presenter received an unrecognised command")

    def _broadcast_selected_workspaces(self):
        self._get_main_presenter().notify_selection_changed()

    @require_main_presenter
    def _get_main_presenter(self):
        return self._main_presenter

    def _load_workspace(self):
        #TODO specify workspace name on load
        #TODO what to do on fail?
        workspace_to_load = self._workspace_manger_view.get_workspace_to_load_path()
        base = os.path.basename(workspace_to_load)
        ws_name = os.path.splitext(base)[0]
        #confirm that user wants to overwrite an existing workspace
        if ws_name in self._work_spaceprovider.get_workspace_names():
            confirm_overwrite = self._workspace_manger_view.confirm_overwrite_workspace()
            if not confirm_overwrite:
                self._workspace_manger_view.no_workspace_has_been_loaded()
                return
        self._work_spaceprovider.load(Filename=workspace_to_load, OutputWorkspace=ws_name)
        self._workspace_manger_view.display_loaded_workspaces(self._work_spaceprovider.get_workspace_names())

    def _save_selected_workspace(self):
        selected_workspaces = self._workspace_manger_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manger_view.error_select_one_workspace()
            return
        if len(selected_workspaces) > 1:
            self._workspace_manger_view.error_select_only_one_workspace()
            return
        selected_workspace = selected_workspaces[0]
        path = self._workspace_manger_view.get_workspace_to_save_filepath()
        self._work_spaceprovider.save_nexus(selected_workspace, path)

    def _remove_selected_workspaces(self):
        selected_workspaces = self._workspace_manger_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manger_view.error_select_one_or_more_workspaces()
            return
        for workspace in selected_workspaces:
            self._work_spaceprovider.delete_workspace(workspace)
        self._workspace_manger_view.display_loaded_workspaces(self._work_spaceprovider.get_workspace_names())

    def _group_selected_workspaces(self):
        selected_workspaces = self._workspace_manger_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manger_view.error_select_one_or_more_workspaces()
            return
        group_name = 'group' + str(self._groupCount)
        self._groupCount += 1
        self._work_spaceprovider.group_workspaces(InputWorkspaces=selected_workspaces, OutputWorkspace=group_name)
        self._workspace_manger_view.display_loaded_workspaces(self._work_spaceprovider.get_workspace_names())

    def _rename_workspace(self):
        selected_workspaces = self._workspace_manger_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manger_view.error_select_one_workspace()
            return
        if len(selected_workspaces) > 1:
            self._workspace_manger_view.error_select_only_one_workspace()
            return
        selected_workspace = selected_workspaces[0]
        new_name = self._workspace_manger_view.get_workspace_new_name()
        self._work_spaceprovider.rename_workspace(selected_workspace, new_name)
        self._workspace_manger_view.display_loaded_workspaces(self._work_spaceprovider.get_workspace_names())

    def get_selected_workspaces(self):
        """Get the currently selected workspaces from the user"""
        return self._workspace_manger_view.get_workspace_selected()

    def update_displayed_workspaces(self):
        """Update the workspaces shown to user.

        This function must be called by the main presenter if any other
        presenter does any operation that changes the name or type of any existing workspace or creates or removes a
        workspace"""
        self._workspace_manger_view.display_loaded_workspaces(self._work_spaceprovider.get_workspace_names())

