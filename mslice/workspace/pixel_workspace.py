from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .histogram_workspace import HistogramWorkspace
from .pixel_mixin import PixelMixin
from .workspace_mixin import WorkspaceOperatorMixin, WorkspaceMixin
from .helperfunctions import attribute_from_log, attribute_to_log

from mantid.api import IMDEventWorkspace
from mantid.simpleapi import DeleteWorkspace



class PixelWorkspace(PixelMixin, WorkspaceOperatorMixin, WorkspaceMixin, WorkspaceBase):
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
        self.axes = []
        if isinstance(mantid_ws, IMDEventWorkspace):
            attribute_from_log(self, mantid_ws)

    def rewrap(self, ws):
        new_ws = PixelWorkspace(ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.e_mode = self.e_mode
        new_ws.e_fixed = self.e_fixed
        new_ws.axes = self.axes
        return new_ws

    def save_attributes(self):
        attrdict = {}
        comstr = self.raw_ws.getComment()
        for k, v in [['comment', comstr], ['axes', self.axes]]:
            if k:
                attrdict[k] = v
        attribute_to_log(attrdict, self.raw_ws)

    def remove_saved_attributes(self):
        attribute_from_log(None, self.raw_ws)

    def __del__(self):
        if hasattr(self, '_raw_ws') and self._raw_ws is not None and self._raw_ws.name().endswith('_HIDDEN'):
            DeleteWorkspace(self._raw_ws)
            self._raw_ws = None
        if hasattr(self, '_histo_ws') and self._histo_ws is not None and self._histo_ws.name().endswith('_HIDDEN'):
            DeleteWorkspace(self._histo_ws)
            self._histo_ws = None
