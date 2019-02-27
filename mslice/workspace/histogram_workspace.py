from __future__ import (absolute_import, division, print_function)
from .base import WorkspaceBase, attribute_from_comment
from .histo_mixin import HistoMixin
from .workspace_mixin import WorkspaceMixin
from .workspace import Workspace

from mantid.api import IMDHistoWorkspace
from mantid.simpleapi import ConvertMDHistoToMatrixWorkspace, Scale, ConvertToDistribution


class HistogramWorkspace(HistoMixin, WorkspaceMixin, WorkspaceBase):
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
        attribute_from_comment(self, mantid_ws)

    def rewrap(self, ws):
        new_ws = HistogramWorkspace(ws, self.name)
        new_ws.is_PSD = self.is_PSD
        new_ws.axes = self.axes
        return new_ws

    def convert_to_matrix(self):
        ws_conv = ConvertMDHistoToMatrixWorkspace(self.name, Normalization='NumEventsNormalization',
                                                  FindXAxis=False, StoreInADS=False, OutputWorkspace=self.name)
        coord = self.get_coordinates()
        bin_size = coord[coord.keys()[0]][1] - coord[coord.keys()[0]][0]
        ws_conv = Scale(ws_conv, bin_size, OutputWorkspace=self.name, StoreInADS=False)
        ConvertToDistribution(ws_conv, StoreInADS=False)
        return Workspace(ws_conv, self.name)
