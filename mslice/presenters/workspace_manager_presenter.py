import os.path

from mslice.widgets.workspacemanager.command import Command
from .interfaces.workspace_manager_presenter import WorkspaceManagerPresenterInterface
from .interfaces.main_presenter import MainPresenterInterface
from .validation_decorators import require_main_presenter

#TODO tell user when file not found,file failed to load

class WorkspaceManagerPresenter(WorkspaceManagerPresenterInterface):
    def __init__(self, workspace_view, workspace_provider):
        #TODO add validation checks
        self._workspace_manger_view = workspace_view
        self._work_spaceprovider = workspace_provider
        self._main_presenter = None

    def register_master(self, main_presenter):
        assert (isinstance(main_presenter, MainPresenterInterface))
        self._main_presenter = main_presenter
        self._main_presenter.register_workspace_selector(self)
        self.update_displayed_workspaces()

    def notify(self, command):
        self._clear_displayed_error()
        if command == Command.LoadWorkspace:
            self._load_workspace()
        elif command == Command.SaveSelectedWorkspace:
            self._save_selected_workspace()
        elif command == Command.RemoveSelectedWorkspaces:
            self._remove_selected_workspaces()
        elif command == Command.RenameWorkspace:
            self._rename_workspace()
        elif command == Command.SelectionChanged:
            self._broadcast_selected_workspaces()
        else:
            raise ValueError("Workspace Manager Presenter received an unrecognised command")

    def _broadcast_selected_workspaces(self):
        self._get_main_presenter().notify_workspace_selection_changed()

    @require_main_presenter
    def _get_main_presenter(self):
        return self._main_presenter

    def _load_workspace(self):
        #TODO specify workspace name on load
        #TODO what to do on fail?
        workspace_to_load = self._workspace_manger_view.get_workspace_to_load_path()
        if not workspace_to_load:
            return
        ws_names = [os.path.splitext(os.path.basename(base))[0] for base in workspace_to_load]
        not_loaded = []
        not_opened = []
        loaded = []
        multi = len(ws_names) > 1
        allChecked = False
        for ii, ws_name in enumerate(ws_names):
            # confirm that user wants to overwrite an existing workspace
            if ws_name in self._work_spaceprovider.get_workspace_names():
                confirm_overwrite = self._workspace_manger_view.confirm_overwrite_workspace()
                if not confirm_overwrite:
                    not_loaded.append(ws_name)
                    continue
            try:
                self._work_spaceprovider.load(filename=workspace_to_load[ii], output_workspace=ws_name)
            except RuntimeError:
                not_opened.append(ws_name)
            else:
                loaded.append(ws_name)
                # Checks if this workspace has efixed set. If not, prompts the user and sets it.
                if self._work_spaceprovider.get_emode(ws_name) == 'Indirect' and not self._work_spaceprovider.has_efixed(ws_name):
                    if not allChecked:
                        Ef, allChecked = self._workspace_manger_view.get_workspace_efixed(ws_name, multi)
                    self._work_spaceprovider.set_efixed(ws_name, Ef)
        if len(not_opened) == len(ws_names):
            self._workspace_manger_view.error_unable_to_open_file()
            return
        elif len(not_opened) > 0:
            errmsg = not_opened[0] if len(not_opened)==1 else ",".join(not_opened)
            self._workspace_manger_view.error_unable_to_open_file(errmsg)
        if len(not_loaded) == len(ws_names):
            self._workspace_manger_view.no_workspace_has_been_loaded()
            return
        elif len(not_loaded) > 0:
            errmsg = not_loaded[0] if len(not_loaded)==1 else ",".join(not_loaded)
            self._workspace_manger_view.no_workspace_has_been_loaded(errmsg)
        self._workspace_manger_view.display_loaded_workspaces(self._work_spaceprovider.get_workspace_names())
        self._workspace_manger_view.set_workspace_selected(
            [self._workspace_manger_view.get_workspace_index(ld_name) for ld_name in loaded])

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
        if not path:
            self._workspace_manger_view.error_invalid_save_path()
            return
        if not path.endswith('.nxs'):
            path += '.nxs'
        try:
            self._work_spaceprovider.save_nexus(selected_workspace, path)
        except RuntimeError:
            self._workspace_manger_view.error_unable_to_save()

    def _remove_selected_workspaces(self):
        selected_workspaces = self._workspace_manger_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manger_view.error_select_one_or_more_workspaces()
            return
        for workspace in selected_workspaces:
            self._work_spaceprovider.delete_workspace(workspace)
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

    def set_selected_workspaces(self, list):
        get_index = self._workspace_manger_view.get_workspace_index
        get_name = self._work_spaceprovider.get_workspace_name
        index_list = []
        for item in list:
            if isinstance(item, basestring):
                index_list.append(get_index(item))
            elif isinstance(item, int):
                index_list.append(item)
            else:
                index_list.append(get_index(get_name(item)))
        self._workspace_manger_view.set_workspace_selected(index_list)

    def get_workspace_provider(self):
        return self._work_spaceprovider

    def update_displayed_workspaces(self):
        """Update the workspaces shown to user.

        This function must be called by the main presenter if any other
        presenter does any operation that changes the name or type of any existing workspace or creates or removes a
        workspace"""
        self._workspace_manger_view.display_loaded_workspaces(self._work_spaceprovider.get_workspace_names())

    def _clear_displayed_error(self):
        self._workspace_manger_view.clear_displayed_error()
