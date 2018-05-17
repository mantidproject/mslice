from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.api import MDNormalization, WorkspaceUnitValidator

from .cut_algorithm import CutAlgorithm

from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, workspace_exists
import mslice.util.mantid.mantid_algorithms as mantid_algorithms
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.workspace import Workspace as Workspace2D
from.cut_normalisation import _num_events_normalized_array

def output_workspace_name(selected_workspace, integration_start, integration_end):
    return selected_workspace + "_cut(" + "{:.3f}".format(integration_start) + "," + "{:.3f}".format(
        integration_end) + ")"

class MantidCutAlgorithm(CutAlgorithm):
    def __init__(self):
        self._converted_nonpsd = None   # Cache for ConvertSpectrumAxis for non-PSD data.

    def compute_cut_xye(self, selected_workspace, cut_axis, integration_axis, is_norm):
        ws_handle = get_workspace_handle(selected_workspace)
        out_ws_name = output_workspace_name(selected_workspace, integration_axis.start, integration_axis.end)

        cut = mantid_algorithms.Cut(OutputWorkspace=out_ws_name, InputWorkspace=ws_handle,
                                    CutAxis=cut_axis.to_dict(), IntegrationAxis=integration_axis.to_dict(),
                                    EMode = ws_handle.e_mode, PSD=ws_handle.is_PSD, NormToOne=is_norm)

        plot_data = _num_events_normalized_array(cut.raw_ws)
        plot_data = plot_data.squeeze()
        with np.errstate(invalid='ignore'):
            if cut.raw_ws.displayNormalization() == MDNormalization.NoNormalization:
                errors = np.sqrt(cut.get_variance())
                errors[np.where(cut.raw_ws.getNumEventsArray() == 0)] = np.nan
            else:
                errors = np.sqrt(cut.get_variance()) / cut.raw_ws.getNumEventsArray()
        errors = errors.squeeze()
        x = np.linspace(cut_axis.start, cut_axis.end, plot_data.size)
        return x, plot_data, errors


    def get_arrays_from_workspace(self, workspace):
        mantid_ws = get_workspace_handle(workspace).raw_ws
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

    def is_cuttable(self, workspace):
        workspace = get_workspace_handle(workspace)
        try:
            is2D = workspace.raw_ws.getNumDims() == 2
        except AttributeError:
            is2D = False
        if not is2D:
            return False
        if isinstance(workspace, PixelWorkspace):
            return True
        else:
            validator = WorkspaceUnitValidator('DeltaE')
            return isinstance(workspace, Workspace2D) and validator.isValid(workspace.raw_ws) == ''

    def set_saved_cut_parameters(self, workspace, axis, parameters):
        if workspace_exists(workspace):
            workspace = get_workspace_handle(workspace)
            workspace.set_cut_params(axis, parameters)

    def get_saved_cut_parameters(self, workspace, axis=None):
        try:
            workspace = get_workspace_handle(workspace)
            if axis is None:
                axis = workspace.cut_params['previous_axis']
            return workspace.cut_params[axis], axis
        except KeyError:
            return None, None

    def is_axis_saved(self, workspace, axis):
        workspace = get_workspace_handle(workspace)
        return workspace.is_axis_saved(axis)

    def _infer_missing_parameters(self, workspace, cut_axis):
        """Infer Missing parameters. This will come in handy at the CLI"""
        assert isinstance(workspace, PixelWorkspace)
        dim = workspace.raw_ws.getDimensionIndexByName(cut_axis.units)
        dim = workspace.raw_ws.getDimension(dim)
        if cut_axis.start is None:
            cut_axis.start = dim.getMinimum()
        if cut_axis.end is None:
            cut_axis.end = dim.getMaximum()
        if cut_axis.step is None:
            cut_axis.step = (cut_axis.end - cut_axis.start)/100
