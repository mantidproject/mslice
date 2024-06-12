from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .histo_mixin import HistoMixin
from .workspace_mixin import WorkspaceOperatorMixin, WorkspaceMixin
from .helperfunctions import attribute_from_log, attribute_to_log, delete_workspace, rename_workspace
from .common_workspace_properties import CommonWorkspaceProperties

from mantid.api import IMDHistoWorkspace

import re


class HistogramWorkspace(HistoMixin, WorkspaceOperatorMixin, WorkspaceMixin, WorkspaceBase, CommonWorkspaceProperties):
    """workspace wrapper for MDHistoWorkspace"""

    def __init__(self, mantid_ws, name):
        if isinstance(mantid_ws, IMDHistoWorkspace):
            self._raw_ws = mantid_ws
        else:
            raise TypeError('HistogramWorkspace expected IMDHistoWorkspace, got %s' % mantid_ws.__class__.__name__)
        CommonWorkspaceProperties.__init__(self)
        self._name = name
        self._cut_params = {}
        self.is_PSD = None
        self.axes = []
        self.norm_to_one = False
        self.parent = None
        self.algorithm = []
        self.intensity_corrected = False
        attribute_from_log(self, mantid_ws)

    @WorkspaceMixin.name.setter
    def name(self, new_name: str):
        raw_name = str(self.raw_ws)
        rename_workspace(raw_name, re.sub(rf"{re.escape(self.name)}\w*", new_name, raw_name))

        self._name = new_name

    def rewrap(self, ws):
        new_ws = HistogramWorkspace(ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.axes = self.axes
        new_ws.is_slice = self.is_slice
        return new_ws

    def convert_to_matrix(self):
        from mslice.util.mantid.mantid_algorithms import ConvertMDHistoToMatrixWorkspace, Scale, ConvertToDistribution
        ws_conv = ConvertMDHistoToMatrixWorkspace(self.name, Normalization='NumEventsNormalization',
                                                  FindXAxis=False, OutputWorkspace='__mat'+self.name)
        coord = self.get_coordinates()
        bin_size = 1
        if self.raw_ws.getNumDims() == 2:
            # for a 2 dimensional workspace use the second dimension to determine bin size
            # this is the case after changing the intensity to GDOS for a cut
            first_dim = coord[self.raw_ws.getDimension(1).name]
        elif self.raw_ws.getNumDims() == 1:
            first_dim = coord[self.raw_ws.getDimension(0).name]
        if len(first_dim) > 1:
            bin_size = first_dim[1] - first_dim[0]
        else:
            raise TypeError('Workspace has only one bin.')
        ws_conv = Scale(ws_conv, bin_size, OutputWorkspace='__mat'+self.name)
        ConvertToDistribution(ws_conv)
        return ws_conv

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
        if hasattr(self, "_raw_ws"):
            delete_workspace(self, self._raw_ws)
