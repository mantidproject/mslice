from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .workspace_mixin import WorkspaceOperatorMixin, WorkspaceMixin
from .helperfunctions import attribute_from_log, attribute_to_log, delete_workspace, rename_workspace
from .common_workspace_properties import CommonWorkspaceProperties

from mantid.api import MatrixWorkspace

import re


class Workspace(WorkspaceOperatorMixin, WorkspaceMixin, WorkspaceBase, CommonWorkspaceProperties):
    """workspace wrapper for MatrixWorkspace"""

    def __init__(self, mantid_ws, name):
        if isinstance(mantid_ws, MatrixWorkspace):
            self._raw_ws = mantid_ws
        else:
            raise TypeError('Workspace expected matrixWorkspace, got %s' % mantid_ws.__class__.__name__)
        CommonWorkspaceProperties.__init__(self)
        self._name = name
        self._cut_params = {}
        self.ef_defined = None
        self.limits = {}
        self.is_PSD = None
        self.e_mode = None
        self.e_fixed = None
        self.axes = []
        attribute_from_log(self, mantid_ws)

    @WorkspaceMixin.name.setter
    def name(self, new_name: str):
        raw_name = str(self.raw_ws)
        rename_workspace(raw_name, re.sub(rf"{self.name}\w*", new_name, raw_name))

        self._name = new_name

    def rewrap(self, mantid_ws):
        new_ws = Workspace(mantid_ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.e_mode = self.e_mode
        new_ws.e_fixed = self.e_fixed
        new_ws.ef_defined = self.ef_defined
        new_ws.limits = self.limits
        new_ws.axes = self.axes
        new_ws.is_slice = self.is_slice
        attribute_from_log(new_ws, mantid_ws)
        return new_ws

    def save_attributes(self):
        attrdict = {}
        for k, v in [['axes', self.axes]]:
            if k:
                attrdict[k] = v
        attribute_to_log(attrdict, self.raw_ws)

    def remove_saved_attributes(self):
        attribute_from_log(None, self.raw_ws)

    def __del__(self):
        if hasattr(self, "_raw_ws"):
            delete_workspace(self, self._raw_ws)
