from __future__ import (absolute_import, division, print_function)
from six import string_types

from .busy import show_busy
from mslice.widgets.workspacemanager.command import Command
from mslice.models.workspacemanager.file_io import get_save_directory
from .interfaces.workspace_manager_presenter import WorkspaceManagerPresenterInterface
from .interfaces.main_presenter import MainPresenterInterface
from .validation_decorators import require_main_presenter



class WorkspaceManagerPresenter(WorkspaceManagerPresenterInterface):
    def __init__(self, workspace_view, workspace_provider):
        # TODO add validation checks
        self._workspace_manager_view = workspace_view
        self._workspace_provider = workspace_provider
        self._main_presenter = None

    def register_master(self, main_presenter):
        assert (isinstance(main_presenter, MainPresenterInterface))
        self._main_presenter = main_presenter
        self._main_presenter.register_workspace_selector(self)
        self.update_displayed_workspaces()

    def notify(self, command):
        self._clear_displayed_error()
        with show_busy(self._workspace_manager_view):
            if command == Command.SaveSelectedWorkspaceNexus:
                self._save_selected_workspace('.nxs')
            elif command == Command.SaveSelectedWorkspaceAscii:
                self._save_selected_workspace('.txt')
            elif command == Command.SaveSelectedWorkspaceMatlab:
                self._save_selected_workspace('.mat')
            elif command == Command.RemoveSelectedWorkspaces:
                self._remove_selected_workspaces()
            elif command == Command.RenameWorkspace:
                self._rename_workspace()
            elif command == Command.CombineWorkspace:
                self._combine_workspace()
            elif command == Command.SelectionChanged:
                self._broadcast_selected_workspaces()
            elif command == Command.Add:
                self._add_workspaces()
            elif command  == Command.Subtract:
                self._subtract_workspace()
            else:
                raise ValueError("Workspace Manager Presenter received an unrecognised command: {}".format(str(command)))

    def _broadcast_selected_workspaces(self):
        self._get_main_presenter().notify_workspace_selection_changed()

    @require_main_presenter
    def _get_main_presenter(self):
        return self._main_presenter

    def change_tab(self, tab):
        self._workspace_manager_view.change_tab(tab)

    def highlight_tab(self, tab):
        self._workspace_manager_view.highlight_tab(tab)

    def _confirm_workspace_overwrite(self, ws_name):
        if ws_name in self._workspace_provider.get_workspace_names():
            return self._workspace_manager_view.confirm_overwrite_workspace()
        else:
            return True

    def _save_selected_workspace(self, extension=None):
        selected_workspaces = self._workspace_manager_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manager_view.error_select_one_workspace()
            return

        try:
            save_directory, save_name, extension = get_save_directory(multiple_files=len(selected_workspaces) > 1,
                                                                      save_as_image=False, default_ext=extension)
        except RuntimeError as e:
            if e.message == "dialog cancelled":
                return
            else:
                raise RuntimeError(e)

        if not save_directory:
            self._workspace_manager_view.error_invalid_save_path()
            return
        try:
            self._workspace_provider.save_workspaces(selected_workspaces, save_directory, save_name, extension)
        except RuntimeError:
            self._workspace_manager_view.error_unable_to_save()

    def _remove_selected_workspaces(self):
        selected_workspaces = self._workspace_manager_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manager_view.error_select_one_or_more_workspaces()
            return
        for workspace in selected_workspaces:
            self._workspace_provider.delete_workspace(workspace)
            self.update_displayed_workspaces()

    def _rename_workspace(self):
        selected_workspaces = self._workspace_manager_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manager_view.error_select_one_workspace()
            return
        if len(selected_workspaces) > 1:
            self._workspace_manager_view.error_select_only_one_workspace()
            return
        selected_workspace = selected_workspaces[0]
        new_name = self._workspace_manager_view.get_workspace_new_name()
        self._workspace_provider.rename_workspace(selected_workspace, new_name)
        self.update_displayed_workspaces()

    def _combine_workspace(self):
        selected_workspaces = self._workspace_manager_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manager_view.error_select_more_than_one_workspaces()
            return
        elif len(selected_workspaces) == 1:
            selected_workspaces.append(str(self._workspace_manager_view.add_workspace_dialog()))
        new_workspace = selected_workspaces[0] + '_combined'
        if all([self._workspace_provider.is_pixel_workspace(workspace) for workspace in selected_workspaces]):
            self._workspace_provider.combine_workspace(selected_workspaces, new_workspace)
        else:
            self._workspace_manager_view.error_select_more_than_one_workspaces()
            return
        self.update_displayed_workspaces()
        return

    def _add_workspaces(self):
        selected_ws = self._workspace_manager_view.get_workspace_selected()
        if not selected_ws:
            self._workspace_manager_view.error_select_one_or_more_workspaces()
            return
        if len(selected_ws) == 1:
            selected_ws.append(self._workspace_manager_view.add_workspace_dialog())
        try:
            self._workspace_provider.add_workspace_runs(selected_ws)
        except ValueError as e:
            self._workspace_manager_view._display_error(str(e))
        self.update_displayed_workspaces()

    def _subtract_workspace(self):
        selected_workspaces = self._workspace_manager_view.get_workspace_selected()
        if not selected_workspaces:
            self._workspace_manager_view.error_select_one_or_more_workspaces()
            return
        try:
            background_ws, ssf  = self._workspace_manager_view.subtraction_input()
        except RuntimeError:
            return
        try:
            self._workspace_provider.subtract(selected_workspaces, background_ws, ssf)
        except ValueError as e:
            self._workspace_manager_view._display_error(str(e))
        self.update_displayed_workspaces()

    def get_selected_workspaces(self):
        """Get the currently selected workspaces from the user"""
        return self._workspace_manager_view.get_workspace_selected()

    def set_selected_workspaces(self, workspace_list):
        get_index = self._workspace_manager_view.get_workspace_index
        get_name = self._workspace_provider.get_workspace_name
        index_list = []
        for item in workspace_list:
            if isinstance(item, string_types):
                index_list.append(get_index(item))
            elif isinstance(item, int):
                index_list.append(item)
            else:
                index_list.append(get_index(get_name(item)))
        self._workspace_manager_view.set_workspace_selected(index_list)

    def get_workspace_provider(self):
        return self._workspace_provider

    def update_displayed_workspaces(self):
        """Update the workspaces shown to user.

        This function must be called by the main presenter if any other
        presenter does any operation that changes the name or type of any existing workspace or creates or removes a
        workspace"""
        self._workspace_manager_view.display_loaded_workspaces(self._workspace_provider.get_workspace_names())

    def _clear_displayed_error(self):
        self._workspace_manager_view.clear_displayed_error()
