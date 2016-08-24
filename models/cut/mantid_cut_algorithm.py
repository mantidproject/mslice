from cut_algorithm import CutAlgorithm
from mantid.simpleapi import BinMD
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from math import floor
import numpy as np


class MantidCutAlgorithm(CutAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def compute_cut(self, selected_workspace, cut_axis, integration_start, integration_end, keepworkspace=False):
        selected_workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        self._infer_missing_parameters(cut_axis)
        n_steps = self._get_number_of_steps(cut_axis)
        cut_binning = " ,".join(map(str, ('|Q|', cut_axis.start, cut_axis.end, n_steps)))
        print ('Warning : disregarding input and binning to 100 bins and integrating along Q')
        integration_binning = "DeltaE," + str(integration_start) + "," + str(integration_end) +",1"
        from random import choice; from string import ascii_lowercase
        output = 'cut_'
        for i in range(4):
            output += choice(ascii_lowercase)
        cut = BinMD(selected_workspace, OutputWorkspace=output, AxisAligned="1", AlignedDim1=integration_binning, AlignedDim0=cut_binning)
        with np.errstate(invalid='ignore'):
            plot_data = cut.getSignalArray() / cut.getNumEventsArray()
        plot_data = plot_data.squeeze()
        plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
        plot_data = plot_data.squeeze()
        x = np.linspace(float(cut_axis.start), float(cut_axis.end), plot_data.size)

        if not keepworkspace:
            self._workspace_provider.delete_workspace(cut)
        return x, plot_data

    def norm(self, x):
        # TODO do something when dividing by zero !!!!!!!
        range_ = x.max() - x.min()
        return (x - x.min())/range_

    def _infer_missing_parameters(self, cut_axis):
        if cut_axis.start is None or cut_axis.end is None:
            raise NotImplementedError("Auto-inference of parameters is not implemented")

    def _get_number_of_steps(self, axis):
        return int(max(1, floor((axis.end - axis.start)/axis.step)))