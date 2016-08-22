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
            x_dim = workspace.getDimension(0)
            y_dim = workspace.getDimension(1)
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
            raise NotImplementedError('Implement Using Rebin2D')
            plot_data = []
            for i in range(workspace.getNumberHistograms()-1,-1,-1):
                plot_data.append(workspace.readY(i))
                x_left = workspace.readX(0)[0]
                x_right = workspace.readX(0)[-1]
                y_top = workspace.getNumberHistograms() - 1
                y_bottom = 0

        return plot_data

    def get_labels(self, workspace, x_axis, y_axis):
        return 'X', 'Y'


    def _get_number_of_steps(self, axis):
        return int(max(1, floor(axis.end - axis.start)/axis.step))

    def _fill_in_missing_input(self,axis,workspace):
        if axis.end is None or axis.start is None or axis.step is None:
            raise NotImplementedError()


