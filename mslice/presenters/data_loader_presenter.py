from __future__ import (absolute_import, division, print_function)

import os

from mslice.presenters.interfaces.data_loader_presenter import DataLoaderPresenterInterface
from mslice.presenters.presenter_utility import PresenterUtility

class DataLoaderPresenter(PresenterUtility, DataLoaderPresenterInterface):


    def __init__(self, data_loader_view):
        self._view = data_loader_view
        self._main_presenter = None
        self._workspace_provider = None

    def set_workspace_provider(self, workspace_provider):
        self._workspace_provider = workspace_provider

    def load_workspace(self, file_paths, merge):
        ws_names = [os.path.splitext(os.path.basename(base))[0] for base in file_paths]
        if merge:
            ws_names = [ws_names[0] + '_merged']
            file_paths = ['+'.join(file_paths)]
        self._load_ws(file_paths, ws_names)

    def _load_ws(self, file_paths, ws_names):
        not_loaded = []
        not_opened = []
        multi = len(ws_names) > 1
        for i, ws_name in enumerate(ws_names):
            if not self._confirm_workspace_overwrite(ws_name):
                not_loaded.append(ws_name)
            else:
                try:
                    self._workspace_provider.load(filename=file_paths[i], output_workspace=ws_name)
                except ValueError as e:
                    self._view.error_loading_workspace(e)
                except RuntimeError:
                    not_opened.append(ws_name)
                else:
                    self.check_efixed(ws_name, multi)
        self._report_load_errors(ws_names, not_opened, not_loaded)
        self._main_presenter.update_displayed_workspaces()

    def check_efixed(self, ws_name, multi=False):
        '''checks if a newly loaded workspace has efixed set'''
        if self._workspace_provider.get_EMode(ws_name) == 'Indirect' and not self._workspace_provider.has_efixed(
                ws_name):
            Ef, allChecked = self._view.get_workspace_efixed(ws_name, multi)
            self._workspace_provider.set_efixed(ws_name, Ef)


    def _confirm_workspace_overwrite(self, ws_name):
        if ws_name in self._workspace_provider.get_workspace_names():
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