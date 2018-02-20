from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.simpleapi import BinMD
from mantid.api import MDNormalization
from mantid.api import IMDEventWorkspace, IMDHistoWorkspace

from .cut_algorithm import CutAlgorithm
from mslice.models.alg_workspace_ops import AlgWorkspaceOps
from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider


class MantidCutAlgorithm(AlgWorkspaceOps, CutAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def compute_cut_xye(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        # TODO Note To reviewer
        # if the is_norm flag is True then _num_events_normalized_array will be called twice, is this OK?
        # Will it cause a significant slowdown on large data? would it be worth caching this?
        cut_computed = False
        copy_created = False
        copy_name = '_to_be_normalized_xyx_123_qme78hj'  # This is just a valid name
        cut = self.compute_cut(selected_workspace, cut_axis, integration_start, integration_end, is_norm=False)
        cut_computed = True
        if is_norm:
            # If the cut previously existed in the ADS we will not modify it
            if not cut_computed:
                copy_created = True
                cut = cut.clone(OutputWorkspace=copy_name)
            self._normalize_workspace(cut)

        plot_data = self._num_events_normalized_array(cut)
        plot_data = plot_data.squeeze()
        with np.errstate(invalid='ignore'):
            errors = np.sqrt(cut.getErrorSquaredArray())/cut.getNumEventsArray()
        errors = errors.squeeze()

        x = np.linspace(cut_axis.start, cut_axis.end, plot_data.size)
        # If the cut already existed in the ADS before this function was called then do not delete it
        if cut_computed:
            self._workspace_provider.delete_workspace(cut)
        if copy_created:
            self._workspace_provider.delete_workspace(copy_name)
        return x, plot_data, errors

    def compute_cut(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        input_workspace_name = selected_workspace
        selected_workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        self._fill_in_missing_input(cut_axis, selected_workspace)
        n_steps = self._get_number_of_steps(cut_axis)
        integration_axis = self.get_other_axis(selected_workspace, cut_axis)

        cut_binning = " ,".join(map(str, (cut_axis.units, cut_axis.start, cut_axis.end, n_steps)))
        integration_binning = integration_axis + "," + str(integration_start) + "," + str(integration_end) + ",1"

        output_ws_name = input_workspace_name + "_cut(" + str(integration_start) + "," + str(integration_end) + ")"
        cut = BinMD(selected_workspace, OutputWorkspace=output_ws_name, AxisAligned="1",
                    AlignedDim1=integration_binning, AlignedDim0=cut_binning)
        if is_norm:
            self._normalize_workspace(cut)
        return cut

    def get_other_axis(self, workspace, axis):
        all_axis = self.get_available_axis(workspace)
        all_axis.remove(axis.units)
        return all_axis[0]

    def is_cuttable(self, workspace):
        workspace = self._workspace_provider.get_workspace_handle(workspace)
        return isinstance(workspace, IMDEventWorkspace) and workspace.getNumDims() == 2

    def set_saved_cut_parameters(self, workspace, axis, parameters):
        self._workspace_provider.setCutParameters(workspace, axis, parameters)

    def get_saved_cut_parameters(self, workspace, axis=None):
        return self._workspace_provider.getCutParameters(workspace, axis)

    def is_axis_saved(self, workspace, axis):
        return self._workspace_provider.isAxisSaved(workspace, axis)

    def _num_events_normalized_array(self, workspace):
        assert isinstance(workspace, IMDHistoWorkspace)
        with np.errstate(invalid='ignore'):
            data = workspace.getSignalArray() / workspace.getNumEventsArray()
        data = np.ma.masked_where(np.isnan(data), data)
        return data

    def _infer_missing_parameters(self, workspace, cut_axis):
        """Infer Missing parameters. This will come in handy at the CLI"""
        assert isinstance(workspace, IMDEventWorkspace)
        dim = workspace.getDimensionIndexByName(cut_axis.units)
        dim = workspace.getDimension(dim)
        if cut_axis.start is None:
            cut_axis.start = dim.getMinimum()
        if cut_axis.end is None:
            cut_axis.end = dim.getMaximum()
        if cut_axis.step is None:
            cut_axis.step = (cut_axis.end - cut_axis.start)/100

    def _normalize_workspace(self, workspace):
        assert isinstance(workspace, IMDHistoWorkspace)
        if workspace.displayNormalization() != MDNormalization.NumEventsNormalization:
            workspace.setDisplayNormalization(MDNormalization.NumEventsNormalization)
        num_events = workspace.getNumEventsArray()
        average_event_intensity = self._num_events_normalized_array(workspace)
        average_event_range = average_event_intensity.max() - average_event_intensity.min()

        normed_average_event_intensity = (average_event_intensity - average_event_intensity.min())/average_event_range
        new_data = normed_average_event_intensity * num_events
        new_data = np.array(new_data)

        new_data = np.nan_to_num(new_data)
        workspace.setSignalArray(new_data)

        errors = workspace.getErrorSquaredArray() / (average_event_range**2)
        workspace.setErrorSquaredArray(errors)
        workspace.setComment("Normalized By MSlice")

    def _was_previously_normalized(self, workspace):
        return workspace.getComment() == "Normalized By MSlice"
