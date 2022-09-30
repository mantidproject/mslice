from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .histogram_workspace import HistogramWorkspace
from .pixel_mixin import PixelMixin
from .workspace_mixin import WorkspaceOperatorMixin, WorkspaceMixin
from .helperfunctions import attribute_from_log, attribute_to_log, delete_workspace, rename_workspace
from .common_workspace_properties import CommonWorkspaceProperties

from mantid.api import IMDEventWorkspace

import re


class PixelWorkspace(PixelMixin, WorkspaceOperatorMixin, WorkspaceMixin, WorkspaceBase, CommonWorkspaceProperties):
    """workspace wrapper for MDEventWorkspace. Converts to HistogramWorkspace internally."""

    def __init__(self, mantid_ws, name):
        """Can be initialized with either MDEventWorkspace or HistogramWorkspace wrapper"""
        if isinstance(mantid_ws, IMDEventWorkspace):
            self._raw_ws = mantid_ws
            self._histo_ws = None
        elif isinstance(mantid_ws, HistogramWorkspace):
            self._raw_ws = None
            self._histo_ws = mantid_ws
        else:
            raise TypeError("PixelWorkspace expected IMDEventWorkspace or HistogramWorkspace, got %s"
                            % mantid_ws.__class__.__name__)
        CommonWorkspaceProperties.__init__(self)
        self._name = name
        self._cut_params = {}
        self.limits = {}
        self.is_PSD = None
        self.e_mode = None
        self.e_fixed = None
        self.axes = []
        if isinstance(mantid_ws, IMDEventWorkspace):
            attribute_from_log(self, mantid_ws)

    @WorkspaceMixin.name.setter
    def name(self, new_name: str):
        if self.raw_ws is not None:
            raw_name = str(self.raw_ws)
            rename_workspace(raw_name, re.sub(rf"{self.name}\w*", new_name, raw_name))
        elif self._histo_ws is not None:
            histo_name = str(self._histo_ws)
            rename_workspace(histo_name, re.sub(rf"{self.name}\w*", new_name, histo_name))

        self._name = new_name

    def rewrap(self, ws):
        new_ws = PixelWorkspace(ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.e_mode = self.e_mode
        new_ws.e_fixed = self.e_fixed
        new_ws.axes = self.axes
        if not new_ws._raw_ws:
            # following binary op div and mul, where _raw_ws not assigned, propagate raw ws. This raw_ws has
            # been in the case of div and mul manipulated as per the binary op.
            new_ws._raw_ws = self.raw_ws
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
        delete_workspace(self, self._raw_ws)
        delete_workspace(self, self._histo_ws)
