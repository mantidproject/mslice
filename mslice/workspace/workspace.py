from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .workspace_mixin import WorkspaceMixin

from mantid.api import MatrixWorkspace


class Workspace(WorkspaceMixin, WorkspaceBase):
    """workspace wrapper for MatrixWorkspace"""

    def __init__(self, mantid_ws):
        if isinstance(mantid_ws, MatrixWorkspace):
            self._raw_ws = mantid_ws
        else:
            raise TypeError('Workspace expected matrixWorkspace, got %s' % mantid_ws.__class__.__name__)

        self.ef_defined = None
        self.limits = {}
        self.cut_parameters = None
        self.is_PSD = None
        self.e_mode = None
        self.e_fixed = None

    def rewrap(self, mantid_ws):
        return Workspace(mantid_ws)

    @property
    def raw_ws(self):
        return self._raw_ws
