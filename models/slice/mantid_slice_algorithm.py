from slice_algorithm import SliceAlgorithm
from mantid.simpleapi import BinMD
from mantid.api import IMDEventWorkspace
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from math import floor
import numpy as np


class MantidSliceAlgorithm(SliceAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def compute_slice(self, selected_workspace, x_axis, y_axis, smoothing):
        workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        if isinstance(workspace,IMDEventWorkspace):
            # TODO implement axis swapping
            # TODO implement input validation and return appropriate error codes
            # Deduct values not supplied by user from workspace
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
            slice = BinMD(InputWorkspace=workspace, AxisAligned="1", AlignedDim0=xbinning, AlignedDim1=ybinning)
            # perform number of events normalization then mask cells where no data was found
            with np.errstate(invalid='ignore'):
                plot_data = slice.getSignalArray() / slice.getNumEventsArray()
            plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
            # The flipud is because mantid plots first row of array at top of plot
            # rot90 switches the x and y axis to to plot what user expected.
            plot_data = np.rot90(plot_data)
            self._workspace_provider.delete_workspace(slice)

        else:
            raise NotImplementedError('Slicing workspace2d is not implemented')
            plot_data = []
            for i in range(workspace.getNumberHistograms()-1,-1,-1):
                plot_data.append(workspace.readY(i))
                x_left = workspace.readX(0)[0]
                x_right = workspace.readX(0)[-1]
                y_top = workspace.getNumberHistograms() - 1
                y_bottom = 0

        return plot_data

    def get_available_axis(self, selected_workspace):
        axis = []
        workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        if isinstance(workspace, IMDEventWorkspace):
            for i in range(workspace.getNumDims()):
                dim_name = workspace.getDimension(i).getName()
                axis.append(dim_name)
        return axis

    def _norm_to_one(self, data):
        data_range = data.max() - data.min()
        return (data - data.min())/data_range

    def _get_number_of_steps(self, axis):
        return int(max(1, floor(axis.end - axis.start)/axis.step))

    def _fill_in_missing_input(self,axis,workspace):
        dim = workspace.getDimensionIndexByName(axis.units)
        dim = workspace.getDimension(dim)

        if axis.start is None:
                axis.start = dim.getMinimum()

        if axis.end is None:
            axis.end = dim.getMaximum()

        if axis.step is None:
            axis.step = (axis.end - axis.start)/100

