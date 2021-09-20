from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase
from .histo_mixin import HistoMixin
from .workspace_mixin import WorkspaceOperatorMixin, WorkspaceMixin
from .helperfunctions import attribute_from_log, attribute_to_log, delete_workspace

from mantid.api import IMDHistoWorkspace


class HistogramWorkspace(HistoMixin, WorkspaceOperatorMixin, WorkspaceMixin, WorkspaceBase):
    """workspace wrapper for MDHistoWorkspace"""

    def __init__(self, mantid_ws, name):
        if isinstance(mantid_ws, IMDHistoWorkspace):
            self._raw_ws = mantid_ws
        else:
            raise TypeError('HistogramWorkspace expected IMDHistoWorkspace, got %s' % mantid_ws.__class__.__name__)
        self.name = name
        self._cut_params = {}
        self.is_PSD = None
        self.axes = []
        self.norm_to_one = False
        self.parent = None
        self.algorithm = []
        attribute_from_log(self, mantid_ws)

    def rewrap(self, ws):
        new_ws = HistogramWorkspace(ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.axes = self.axes
        return new_ws

    def convert_to_matrix(self):
        from mslice.util.mantid.mantid_algorithms import ConvertMDHistoToMatrixWorkspace, Scale, ConvertToDistribution
        ws_conv = ConvertMDHistoToMatrixWorkspace(self.name, Normalization='NumEventsNormalization',
                                                  FindXAxis=False, OutputWorkspace='__mat'+self.name)
        coord = self.get_coordinates()
        first_dim = coord[self.raw_ws.getDimension(0).getName()]
        bin_size = first_dim[1] - first_dim[0]
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
        delete_workspace(self, self._raw_ws)
