from __future__ import (absolute_import, division, print_function)

from mantid.api import WorkspaceUnitValidator

from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.mantid import mantid_algorithms
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.workspace import Workspace as Workspace2D


def output_workspace_name(selected_workspace: str, integration_start: float, integration_end: float) -> str:
    return f"{selected_workspace}_cut({integration_start:.3f},{integration_end:.3f})"


def compute_cut(workspace, cut_axis, integration_axis, is_norm, algo='Rebin', store=True):
    out_ws_name = output_workspace_name(workspace.name, integration_axis.start, integration_axis.end)
    cut = mantid_algorithms.Cut(OutputWorkspace=_make_name_unique(out_ws_name), store=store, InputWorkspace=workspace,
                                CutAxis=cut_axis.to_dict(), IntegrationAxis=integration_axis.to_dict(),
                                EMode=workspace.e_mode, PSD=workspace.is_PSD, NormToOne=is_norm,
                                Algorithm=algo)
    cut.parent = workspace.name
    return cut


def _make_name_unique(ws_name, i=1):
    try:
        get_workspace_handle(ws_name)
        if i == 1:
            ws_name = ws_name + f"_({i})"
        else:
            ws_name = ws_name.replace(f"_({i-1})",  f"_({i})")
        i += 1
        ws_name = _make_name_unique(ws_name, i)
    except KeyError:
        pass
    return ws_name


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
