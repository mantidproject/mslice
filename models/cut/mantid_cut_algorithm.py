from cut_algorithm import CutAlgorithm
from mantid.simpleapi import BinMD
from mantid.api import IMDEventWorkspace, IMDHistoWorkspace
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from math import floor
import numpy as np


class MantidCutAlgorithm(CutAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def compute_cut_xy(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        cut = self.compute_cut(selected_workspace, cut_axis, integration_start, integration_end, is_norm)

        plot_data = self._num_events_normalized_array(cut)
        plot_data = plot_data.squeeze()

        x = np.linspace(cut_axis.start, cut_axis.end, plot_data.size)
        self._workspace_provider.delete_workspace(cut)

        return x, plot_data

    def compute_cut(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        # TODO Note To reviewer
        # if the is_norm flag is True then _num_events_normalized_array will be called twice, is this OK?
        # Will it cause a significant slowdown of on large data? would it be worth caching this?
        selected_workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        self._infer_missing_parameters(selected_workspace, cut_axis)
        n_steps = self._get_number_of_steps(cut_axis)

        axis = self.get_available_axis(selected_workspace)
        axis.remove(cut_axis.units)
        integration_axis = axis[0]

        cut_binning = " ,".join(map(str, (cut_axis.units, cut_axis.start, cut_axis.end, n_steps)))
        integration_binning = integration_axis + "," + str(integration_start) + "," + str(integration_end) + ",1"
        from random import choice; from string import ascii_lowercase
        output = 'cut_'
        for i in range(4):
            output += choice(ascii_lowercase)
        cut = BinMD(selected_workspace, OutputWorkspace=output, AxisAligned="1", AlignedDim1=integration_binning,
                    AlignedDim0=cut_binning)
        if is_norm:
            self._normalize_workspace(cut)
        return cut

    def _normalize_workspace(self, workspace):
        assert isinstance(workspace, IMDHistoWorkspace)
        num_events = workspace.getNumEventsArray()
        average_event_intensity = self._num_events_normalized_array(workspace)
        average_event_range = average_event_intensity.max() - average_event_intensity.min()

        normed_average_event_intensity = (average_event_intensity - average_event_intensity.min())/average_event_range
        new_data = normed_average_event_intensity * num_events
        new_data = np.array(new_data)

        new_data = np.nan_to_num(new_data)
        workspace.setSignalArray(new_data)


    def get_available_axis(self, workspace):
        if isinstance(workspace, str):
            workspace = self._workspace_provider.get_workspace_handle(workspace)
        if not isinstance(workspace, IMDEventWorkspace):
            return []
        dim_names = []
        for i in range(workspace.getNumDims()):
            dim_names.append(workspace.getDimension(i).getName())
        return dim_names

    def _num_events_normalized_array(self, workspace):
        assert isinstance(workspace, IMDHistoWorkspace)
        with np.errstate(invalid='ignore'):
            data = workspace.getSignalArray() / workspace.getNumEventsArray()
        data = np.ma.masked_where(np.isnan(data), data)
        return data

    def _infer_missing_parameters(self, workspace, cut_axis):
        assert isinstance(workspace, IMDEventWorkspace)
        dim = workspace.getDimensionIndexByName(cut_axis.units)
        dim = workspace.getDimension(dim)
        if cut_axis.start is None:
            cut_axis.start = dim.getMinimum()
        if cut_axis.end is None:
            cut_axis.end = dim.getMaximum()
        if cut_axis.step is None:
            cut_axis.step = (cut_axis.end - cut_axis.start)/100

    def _get_number_of_steps(self, axis):
        return int(max(1, floor((axis.end - axis.start)/axis.step)))