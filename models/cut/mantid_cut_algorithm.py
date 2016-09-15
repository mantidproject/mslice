from cut_algorithm import CutAlgorithm
from mantid.simpleapi import BinMD
from mantid.api import MDNormalization
from mantid.api import IMDEventWorkspace, IMDHistoWorkspace
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from presenters.slice_plotter_presenter import Axis
from math import floor
import numpy as np


class MantidCutAlgorithm(CutAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def compute_cut_xye(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        # TODO Note To reviewer
        # if the is_norm flag is True then _num_events_normalized_array will be called twice, is this OK?
        # Will it cause a significant slowdown on large data? would it be worth caching this?
        cut_computed = False
        copy_created = False
        copy_name = '_to_be_normalized_xyx_123_qme78hj'  # This is just a valid name
        if not self.is_cut(selected_workspace):
            cut = self.compute_cut(selected_workspace, cut_axis, integration_start, integration_end, is_norm=False)
            cut_computed = True
        else:
            cut = self._workspace_provider.get_workspace_handle(selected_workspace)
        if is_norm:
            # If the cut previously existed in the ADS we will not modify it
            if not cut_computed:
                copy_created = True
                cut = cut.clone(OutputWorkspace=copy_name)
            self._normalize_workspace(cut)

        plot_data = self._num_events_normalized_array(cut)
        plot_data = plot_data.squeeze()
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

        output_workspace_name = input_workspace_name + "_cut"
        index = 1
        existing_workspaces = self._workspace_provider.get_workspace_names()
        while output_workspace_name + str(index) in existing_workspaces:
            index += 1
        output_workspace_name += str(index)
        cut = BinMD(selected_workspace, OutputWorkspace=output_workspace_name, AxisAligned="1", AlignedDim1=integration_binning,
                    AlignedDim0=cut_binning)
        if is_norm:
            self._normalize_workspace(cut)
        return cut

    def get_available_axis(self, workspace):
        if isinstance(workspace, str):
            workspace = self._workspace_provider.get_workspace_handle(workspace)
        dim_names = []
        for i in range(workspace.getNumDims()):
            dim_names.append(workspace.getDimension(i).getName())
        return dim_names

    def get_other_axis(self, workspace, axis):
        all_axis = self.get_available_axis(workspace)
        all_axis.remove(axis.units)
        return all_axis[0]

    def is_cut(self, workspace):
        workspace = self._workspace_provider.get_workspace_handle(workspace)
        if not isinstance(workspace, IMDHistoWorkspace):
            return False
        if workspace.getNumDims() != 2 or len(workspace.getNonIntegratedDimensions()) != 1:
            return False
        history = workspace.getHistory()
        # If any of these were set in the last BinMD call then this is not a cut
        empty_params = ('AlignedDim2','AlignedDim3', 'AlignedDim4', 'AlignedDim5')
        # If these were not set then this is not a cut
        used_params = ('AxisAligned', 'AlignedDim0', 'AlignedDim1')
        for i in range(history.size()-1, -1, -1):
            try:
                algorithm = history.getAlgorithm(i)
            except RuntimeError:
                return False
            if algorithm.name() == 'BinMD':
                if all(map(lambda x: not algorithm.getPropertyValue(x), empty_params))\
                        and all(map(lambda x: algorithm.getPropertyValue(x), used_params)):
                    return True
                break
        return False

    def is_cuttable(self, workspace):
        workspace = self._workspace_provider.get_workspace_handle(workspace)
        return isinstance(workspace, IMDEventWorkspace) and workspace.getNumDims() == 2

    def get_cut_params(self, cut_workspace):
        cut_workspace = self._workspace_provider.get_workspace_handle(cut_workspace)
        assert isinstance(cut_workspace, IMDHistoWorkspace)
        is_norm = self._was_previously_normalized(cut_workspace)
        bin_md = None
        history = cut_workspace.getHistory()
        for i in range(history.size()-1,-1,-1):
            algorithm = history.getAlgorithm(i)
            if algorithm.name() == 'BinMD':
                bin_md = algorithm
                break
        integration_binning = bin_md.getPropertyValue("AlignedDim1").split(",")
        integration_range = float(integration_binning[1]), float(integration_binning[2])

        cut_binning = bin_md.getPropertyValue("AlignedDim0").split(",")
        # The axis name is retreived from the workspace directly and not the binning string to avoid
        # adding/removing trailing spaces
        dim_name = cut_workspace.getDimension(0).getName()
        cut_axis = Axis(dim_name, *map(float,cut_binning[1:]))
        cut_axis.step = (cut_axis.end - cut_axis.start)/float(cut_binning[-1])
        return cut_axis, integration_range, is_norm

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
        assert workspace.displayNormalization() == MDNormalization.NumEventsNormalization
        num_events = workspace.getNumEventsArray()
        average_event_intensity = self._num_events_normalized_array(workspace)
        average_event_range = average_event_intensity.max() - average_event_intensity.min()

        normed_average_event_intensity = (average_event_intensity - average_event_intensity.min())/average_event_range
        new_data = normed_average_event_intensity * num_events
        new_data = np.array(new_data)

        new_data = np.nan_to_num(new_data)
        workspace.setSignalArray(new_data)

        errors = workspace.getErrorSquaredArray() /(average_event_range**2)
        workspace.setErrorSquaredArray(errors)
        workspace.setComment("Normalized By MSlice")

    def _get_number_of_steps(self, axis):
        return int(max(1, floor((axis.end - axis.start)/axis.step)))

    def _was_previously_normalized(self, workspace):
        return workspace.getComment() == "Normalized By MSlice"

    def _fill_in_missing_input(self,axis,workspace):
        dim = workspace.getDimensionIndexByName(axis.units)
        dim = workspace.getDimension(dim)

        if axis.start is None:
                axis.start = dim.getMinimum()

        if axis.end is None:
            axis.end = dim.getMaximum()

        if axis.step is None:
            axis.step = (axis.end - axis.start)/100