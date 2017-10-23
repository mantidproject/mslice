from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.simpleapi import BinMD
from mantid.api import IMDEventWorkspace

from .slice_algorithm import SliceAlgorithm
from mslice.models.alg_workspace_ops import AlgWorkspaceOps
from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider


class MantidSliceAlgorithm(AlgWorkspaceOps, SliceAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def compute_slice(self, selected_workspace, x_axis, y_axis, smoothing, norm_to_one):
        workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        assert isinstance(workspace,IMDEventWorkspace)

        self._fill_in_missing_input(x_axis, workspace)
        self._fill_in_missing_input(y_axis, workspace)

        n_x_bins = self._get_number_of_steps(x_axis)
        n_y_bins = self._get_number_of_steps(y_axis)
        x_dim_id = workspace.getDimensionIndexByName(x_axis.units)
        y_dim_id = workspace.getDimensionIndexByName(y_axis.units)
        x_dim = workspace.getDimension(x_dim_id)
        y_dim = workspace.getDimension(y_dim_id)
        xbinning = x_dim.getName() + "," + str(x_axis.start) + "," + str(x_axis.end) + "," + str(n_x_bins)
        ybinning = y_dim.getName() + "," + str(y_axis.start) + "," + str(y_axis.end) + "," + str(n_y_bins)
        thisslice = BinMD(InputWorkspace=workspace, AxisAligned="1", AlignedDim0=xbinning, AlignedDim1=ybinning)
        # perform number of events normalization then mask cells where no data was found
        with np.errstate(invalid='ignore'):
            plot_data = thisslice.getSignalArray() / thisslice.getNumEventsArray()
        plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
        # rot90 switches the x and y axis to to plot what user expected.
        plot_data = np.rot90(plot_data)
        self._workspace_provider.delete_workspace(thisslice)
        boundaries = [x_axis.start, x_axis.end, y_axis.start, y_axis.end]
        if norm_to_one:
            plot_data = self._norm_to_one(plot_data)
        return plot_data, boundaries

    def _norm_to_one(self, data):
        data_range = data.max() - data.min()
        return (data - data.min())/data_range
