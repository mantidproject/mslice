from __future__ import (absolute_import, division, print_function)

import os

from .busy import show_busy
from mslice.models.workspacemanager.workspace_algorithms import load
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, get_visible_workspace_names
from mslice.presenters.interfaces.data_loader_presenter import DataLoaderPresenterInterface
from mslice.presenters.presenter_utility import PresenterUtility
from mslice.models.workspacemanager.file_io import load_from_ascii


class DataLoaderPresenter(PresenterUtility, DataLoaderPresenterInterface):

    def __init__(self, data_loader_view):
        self._view = data_loader_view
        self._main_presenter = None
        self._EfCache = None

    def load_workspace(self, file_paths, merge=False):
        """
        Loads one or more workspaces.
        :param file_paths: list of paths to files to load
        :param merge: boolean - whether to combine files into a single workspace
        """
        with show_busy(self._view):
            ws_names = [os.path.splitext(os.path.basename(base))[0] for base in file_paths]
            if merge:
                if not self.file_types_match(file_paths):
                    self._view.error_merge_different_file_formats()
                    return
                ws_names = [ws_names[0] + '_merged']
                file_paths = ['+'.join(file_paths)]
            self._load_ws(file_paths, ws_names)

    def _load_ws(self, file_paths, ws_names):
        not_loaded = []
        not_opened = []
        multi = len(ws_names) > 1
        allChecked = False
        for i, ws_name in enumerate(ws_names):
            if not self._confirm_workspace_overwrite(ws_name):
                not_loaded.append(ws_name)
            else:
                try:
                    if file_paths[i].endswith('.txt'):
                        load_from_ascii(file_paths[i], ws_name)
                    else:
                        load(filename=file_paths[i], output_workspace=ws_name)
                except ValueError as e:
                    self._view.error_loading_workspace(e)
                except RuntimeError:
                    not_opened.append(ws_name)
                else:
                    if not allChecked:
                        allChecked = self.check_efixed(ws_name, multi)
                    else:
                        get_workspace_handle(ws_name).e_fixed = self._EfCache
                    self._main_presenter.show_workspace_manager_tab()
                    self._main_presenter.update_displayed_workspaces()
                    self._main_presenter.show_tab_for_workspace(get_workspace_handle(ws_name))
        self._report_load_errors(ws_names, not_opened, not_loaded)

    def file_types_match(self, selected_files):
        extensions = [selection.rsplit('.', 1)[-1] for selection in selected_files]
        return all(ext == extensions[0] for ext in extensions)

    def check_efixed(self, ws_name, multi=False):
        '''checks if a newly loaded workspace has efixed set'''
        ws = get_workspace_handle(ws_name)
        if ws.e_mode == 'Indirect' and not ws.ef_defined:
            Ef, allChecked = self._view.get_workspace_efixed(ws_name, multi, self._EfCache)
            self._EfCache = Ef
            ws.e_fixed = Ef
            return allChecked

    def _confirm_workspace_overwrite(self, ws_name):
        if ws_name in get_visible_workspace_names():
            return self._view.confirm_overwrite_workspace()
        else:
            return True

    def _report_load_errors(self, ws_names, not_opened, not_loaded):
        if len(not_opened) == len(ws_names):
            self._view.error_unable_to_open_file()
            return
        elif len(not_opened) > 0:
            errmsg = not_opened[0] if len(not_opened) == 1 else ",".join(not_opened)
            self._view.error_unable_to_open_file(errmsg)
        if len(not_loaded) == len(ws_names):
            self._view.no_workspace_has_been_loaded()
            return
        elif len(not_loaded) > 0:
            errmsg = not_loaded[0] if len(not_loaded) == 1 else ",".join(not_loaded)
            self._view.no_workspace_has_been_loaded(errmsg)

    def workspace_selection_changed(self):
        pass
