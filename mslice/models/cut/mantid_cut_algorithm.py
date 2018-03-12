from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.simpleapi import BinMD, SofQW3, Rebin2D, ConvertSpectrumAxis, CreateMDHistoWorkspace
from mantid.dataobjects import Workspace2D
from mantid.api import MDNormalization, IMDEventWorkspace, IMDHistoWorkspace, WorkspaceUnitValidator

from .cut_algorithm import CutAlgorithm
from mslice.models.alg_workspace_ops import AlgWorkspaceOps
from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider


def output_workspace_name(selected_workspace, integration_start, integration_end):
    return selected_workspace + "_cut(" + "{:.3f}".format(integration_start) + "," + "{:.3f}".format(
        integration_end) + ")"


class MantidCutAlgorithm(AlgWorkspaceOps, CutAlgorithm):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()
        self._converted_nonpsd = None   # Cache for ConvertSpectrumAxis for non-PSD data.

    def compute_cut_xye(self, selected_workspace, cut_axis, integration_axis, is_norm):
        # TODO Note To reviewer
        # if the is_norm flag is True then _num_events_normalized_array will be called twice, is this OK?
        # Will it cause a significant slowdown on large data? would it be worth caching this?
        cut_computed = False
        copy_created = False
        copy_name = '_to_be_normalized_xyx_123_qme78hj'  # This is just a valid name
        cut = self.compute_cut(selected_workspace, cut_axis, integration_axis, is_norm=False)
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
            if cut.displayNormalization() == MDNormalization.NoNormalization:
                errors = np.sqrt(cut.getErrorSquaredArray())
                errors[np.where(cut.getNumEventsArray() == 0)] = np.nan
            else:
                errors = np.sqrt(cut.getErrorSquaredArray()) / cut.getNumEventsArray()
        errors = errors.squeeze()

        x = np.linspace(cut_axis.start, cut_axis.end, plot_data.size)
        # If the cut already existed in the ADS before this function was called then do not delete it
        if cut_computed:
            self._workspace_provider.delete_workspace(cut)
        if copy_created:
            self._workspace_provider.delete_workspace(copy_name)
        return x, plot_data, errors

    def compute_cut(self, selected_workspace, cut_axis, integration_axis, is_norm):
        input_workspace_name = selected_workspace
        out_ws_name = output_workspace_name(selected_workspace, integration_axis.start, integration_axis.end)
        selected_workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        integration_start = integration_axis.start
        integration_end = integration_axis.end
        integration_units = integration_axis.units

        if self._workspace_provider.is_PSD(input_workspace_name):
            cut = self._compute_cut_PSD(input_workspace_name, out_ws_name, selected_workspace, cut_axis,
                                        integration_start, integration_end)
        else:
            cut = self._compute_cut_nonPSD(input_workspace_name, out_ws_name, selected_workspace, cut_axis,
                                           integration_start, integration_end, integration_units)

        if is_norm:
            self._normalize_workspace(cut)
        return cut

    def _compute_cut_PSD(self, input_workspace_name, out_ws_name, selected_workspace, cut_axis,
                         integration_start, integration_end):
        self._fill_in_missing_input(cut_axis, selected_workspace)
        n_steps = self._get_number_of_steps(cut_axis)
        integration_axis = self.get_other_axis(selected_workspace, cut_axis)

        cut_binning = " ,".join(map(str, (cut_axis.units, cut_axis.start, cut_axis.end, n_steps)))
        integration_binning = integration_axis + "," + str(integration_start) + "," + str(integration_end) + ",1"

        cut = BinMD(selected_workspace, OutputWorkspace=out_ws_name, AxisAligned="1",
                    AlignedDim1=integration_binning, AlignedDim0=cut_binning)
        return cut

    def _compute_cut_nonPSD(self, input_workspace_name, out_ws_name, selected_workspace, cut_axis,
                            integration_start, integration_end, integration_units):
        cut_binning = " ,".join(map(str, (cut_axis.start, cut_axis.step, cut_axis.end)))
        int_binning = " ,".join(map(str, (integration_start, integration_end - integration_start, integration_end)))
        emode = self._workspace_provider.get_EMode(input_workspace_name)
        if self._converted_nonpsd and self._converted_nonpsd[0] != input_workspace_name:
            self._converted_nonpsd = None
        if cut_axis.units == '|Q|':
            SofQW3(selected_workspace, OutputWorkspace=out_ws_name, EMode=emode,
                   QAxisBinning=cut_binning, EAxisBinning=int_binning)
            idx = 1
            unit = 'MomentumTransfer'
            name = '|Q|'
        elif cut_axis.units == 'Degrees':
            if not self._converted_nonpsd:
                self._converted_nonpsd = (input_workspace_name,
                                          ConvertSpectrumAxis(selected_workspace, Target='theta',
                                                              OutputWorkspace='__convToTheta', StoreInADS=False))
            Rebin2D(self._converted_nonpsd[1], OutputWorkspace=out_ws_name, Axis1Binning=int_binning, Axis2Binning=cut_binning)
            idx = 1
            unit = 'Degrees'
            name = 'Theta'
        elif integration_units == '|Q|':
            SofQW3(selected_workspace, OutputWorkspace=out_ws_name, EMode=emode,
                   QAxisBinning=int_binning, EAxisBinning=cut_binning)
            idx = 0
            unit = 'DeltaE'
            name = 'EnergyTransfer'
        else:
            if not self._converted_nonpsd:
                self._converted_nonpsd = (input_workspace_name,
                                          ConvertSpectrumAxis(selected_workspace, Target='theta',
                                                              OutputWorkspace='__convToTheta', StoreInADS=False))
            Rebin2D(self._converted_nonpsd[1], OutputWorkspace=out_ws_name, Axis1Binning=cut_binning, Axis2Binning=int_binning)
            idx = 0
            unit = 'DeltaE'
            name = 'EnergyTransfer'
        ws_out = self._workspace_provider.get_workspace_handle(out_ws_name)
        xdim = ws_out.getDimension(idx)
        extents = " ,".join(map(str, (xdim.getMinimum(), xdim.getMaximum())))
        cut = CreateMDHistoWorkspace(OutputWorkspace=out_ws_name, SignalInput=ws_out.extractY(), ErrorInput=ws_out.extractE(),
                                     Dimensionality=1, Extents=extents, NumberOfBins=xdim.getNBins(), Names=name, Units=unit)
        return cut

    def get_arrays_from_workspace(self, workspace):
        mantid_ws = self._workspace_provider.get_workspace_handle(workspace)
        dim = mantid_ws.getDimension(0)
        x = np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins())
        with np.errstate(invalid='ignore'):
            if mantid_ws.displayNormalization() == MDNormalization.NoNormalization:
                y = np.array(mantid_ws.getSignalArray())
                e = np.sqrt(mantid_ws.getErrorSquaredArray())
                nanidx = np.where(mantid_ws.getNumEventsArray() == 0)
                y[nanidx] = np.nan
                e[nanidx] = np.nan
            else:
                y = mantid_ws.getSignalArray() / mantid_ws.getNumEventsArray()
                e = np.sqrt(mantid_ws.getErrorSquaredArray()) / mantid_ws.getNumEventsArray()
        e = e.squeeze()
        return x, y, e, dim.getUnits()

    def get_other_axis(self, workspace, axis):
        all_axis = self.get_available_axis(workspace)
        all_axis.remove(axis.units)
        return all_axis[0]

    def is_cuttable(self, workspace):
        workspace = self._workspace_provider.get_workspace_handle(workspace)
        try:
            is2D = workspace.getNumDims() == 2
        except AttributeError:
            is2D = False
        if not is2D:
            return False
        if isinstance(workspace, IMDEventWorkspace):
            return True
        else:
            validator = WorkspaceUnitValidator('DeltaE')
            return isinstance(workspace, Workspace2D) and validator.isValid(workspace) == ''

    def set_saved_cut_parameters(self, workspace, axis, parameters):
        self._workspace_provider.setCutParameters(workspace, axis, parameters)

    def get_saved_cut_parameters(self, workspace, axis=None):
        return self._workspace_provider.getCutParameters(workspace, axis)

    def is_axis_saved(self, workspace, axis):
        return self._workspace_provider.isAxisSaved(workspace, axis)

    def _num_events_normalized_array(self, workspace):
        assert isinstance(workspace, IMDHistoWorkspace)
        with np.errstate(invalid='ignore'):
            if workspace.displayNormalization() == MDNormalization.NoNormalization:
                data = np.array(workspace.getSignalArray())
                data[np.where(workspace.getNumEventsArray() == 0)] = np.nan
            else:
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
        num_events = workspace.getNumEventsArray()
        average_event_intensity = self._num_events_normalized_array(workspace)
        average_event_range = average_event_intensity.max() - average_event_intensity.min()

        normed_average_event_intensity = (average_event_intensity - average_event_intensity.min())/average_event_range
        if workspace.displayNormalization() == MDNormalization.NoNormalization:
            new_data = normed_average_event_intensity
        else:
            new_data = normed_average_event_intensity * num_events
        new_data = np.array(new_data)

        new_data = np.nan_to_num(new_data)
        workspace.setSignalArray(new_data)

        errors = workspace.getErrorSquaredArray() / (average_event_range**2)
        workspace.setErrorSquaredArray(errors)
        workspace.setComment("Normalized By MSlice")

    def _was_previously_normalized(self, workspace):
        return workspace.getComment() == "Normalized By MSlice"
