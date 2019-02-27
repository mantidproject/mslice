from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase, attribute_from_comment
from .workspace_mixin import WorkspaceMixin

from mantid.api import MatrixWorkspace


class Workspace(WorkspaceMixin, WorkspaceBase):
    """workspace wrapper for MatrixWorkspace"""

    def __init__(self, mantid_ws, name):
        if isinstance(mantid_ws, MatrixWorkspace):
            self._raw_ws = mantid_ws
        else:
            raise TypeError('Workspace expected matrixWorkspace, got %s' % mantid_ws.__class__.__name__)
        self.name = name
        self._cut_params = {}
        self.ef_defined = None
        self.limits = {}
        self.is_PSD = None
        self.e_mode = None
        self.e_fixed = None
        self.axes = []
        attribute_from_comment(self, mantid_ws)

    def rewrap(self, mantid_ws):
        new_ws = Workspace(mantid_ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.e_mode = self.e_mode
        new_ws.e_fixed = self.e_fixed
        new_ws.ef_defined = self.ef_defined
        new_ws.limits = self.limits
        new_ws.axes = self.axes
        return new_ws
