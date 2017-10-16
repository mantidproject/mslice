from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .histo_mixin import HistoMixin
from .workspace_mixin import WorkspaceMixin

from mantid.api import IMDHistoWorkspace


class HistogramWorkspace(HistoMixin, WorkspaceMixin, WorkspaceBase):
    """workspace wrapper for MDHistoWorkspace"""

    def __init__(self, mantid_ws):
        if isinstance(mantid_ws, IMDHistoWorkspace):
            self._raw_ws = mantid_ws
        else:
            raise TypeError('HistogramWorkspace expected IMDHistoWorkspace, got %s' % mantid_ws.__class__.__name__)

    def rewrap(self, ws):
        return HistogramWorkspace(ws)
