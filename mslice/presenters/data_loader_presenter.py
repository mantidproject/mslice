from __future__ import (absolute_import, division, print_function)

import os

from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from mslice.presenters.presenter_utility import PresenterUtility

class DataLoaderPresenter(PresenterUtility): #TODO: create interface


    def __init__(self, data_loader_view):
        self._view = data_loader_view
        self._main_presenter = None
        self._workspace_provider = None

    def set_workspace_provider(self, workspace_provider):
        self._workspace_provider = workspace_provider

    def load_workspace(self, workspace_to_load):
        ws_names = [os.path.splitext(os.path.basename(base))[0] for base in workspace_to_load]
        not_loaded = []
        not_opened = []
        loaded = []
        multi = len(ws_names) > 1
        allChecked = False
        for ii, ws_name in enumerate(ws_names):
            # confirm that user wants to overwrite an existing workspace
            if not self._confirm_workspace_overwrite(ws_name):
                not_loaded.append(ws_name)
                continue
            try:
                self._workspace_provider.load(filename=workspace_to_load[ii], output_workspace=ws_name)
            except RuntimeError:
                not_opened.append(ws_name)
            else:
                loaded.append(ws_name)
                # Checks if this workspace has efixed set. If not, prompts the user and sets it.
                if self._workspace_provider.get_EMode(ws_name) == 'Indirect' and not self._workspace_provider.has_efixed(ws_name):
                    if not allChecked:
                        Ef, allChecked = self._view.get_workspace_efixed(ws_name, multi)
                    self._workspace_provider.set_efixed(ws_name, Ef)

        self._report_load_errors(ws_names, not_opened, not_loaded)
        self._main_presenter.update_displayed_workspaces()

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