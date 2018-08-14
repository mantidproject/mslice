from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.api import MDNormalization, WorkspaceUnitValidator

from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.mantid import run_algorithm
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.workspace import Workspace as Workspace2D


def output_workspace_name(selected_workspace, integration_start, integration_end):
    return selected_workspace + "_cut(" + "{:.3f}".format(integration_start) + "," + "{:.3f}".format(
        integration_end) + ")"


def compute_cut(workspace, cut_axis, integration_axis, is_norm, store=True):
    out_ws_name = output_workspace_name(workspace.name, integration_axis.start, integration_axis.end)
    cut = run_algorithm('Cut', output_name=out_ws_name, store=store, InputWorkspace=workspace,
                        CutAxis=cut_axis.to_dict(), IntegrationAxis=integration_axis.to_dict(),
                        EMode = workspace.e_mode, PSD=workspace.is_PSD, NormToOne=is_norm)
    return cut


def get_arrays_from_workspace(workspace):
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


def is_cuttable(workspace):
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


def _infer_missing_parameters(workspace, cut_axis):
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
