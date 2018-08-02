from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .histogram_workspace import HistogramWorkspace
from .pixel_mixin import PixelMixin
from .workspace_mixin import WorkspaceMixin

from mantid.api import IMDEventWorkspace



class PixelWorkspace(PixelMixin, WorkspaceMixin, WorkspaceBase):
    """workspace wrapper for MDEventWorkspace. Converts to HistogramWorkspace internally."""

    def __init__(self, mantid_ws, name):
        """Can be initialized with either MDEventWorkspace or HistogramWorkspace wrapper"""
        if isinstance(mantid_ws, IMDEventWorkspace):
            self._raw_ws = mantid_ws
            self._histo_ws = None
        elif isinstance(mantid_ws, HistogramWorkspace):
            self._histo_ws = mantid_ws
        else:
            raise TypeError("PixelWorkspace expected IMDEventWorkspace or HistogramWorkspace, got %s"
                            % mantid_ws.__class__.__name__)
        self.name = name
        self._cut_params = {}
        self.limits = {}
        self.is_PSD = None
        self.e_mode = None
        self.e_fixed = None

    def rewrap(self, ws):
        new_ws = PixelWorkspace(ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.e_mode = self.e_mode
        new_ws.e_fixed = self.e_fixed
        return new_ws
