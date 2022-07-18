from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.api import WorkspaceUnitValidator
from scipy import constants

from mslice.models.labels import is_momentum, is_twotheta
from mslice.models.workspacemanager.workspace_algorithms import propagate_properties
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.mantid import mantid_algorithms

from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.workspace import Workspace


def compute_slice(selected_workspace, x_axis, y_axis, norm_to_one):
    workspace = get_workspace_handle(selected_workspace)
    slice = mantid_algorithms.Slice(OutputWorkspace='__' + workspace.name, InputWorkspace=workspace,
                                    XAxis=x_axis.to_dict(), YAxis=y_axis.to_dict(), PSD=workspace.is_PSD,
                                    EMode=workspace.e_mode, NormToOne=norm_to_one)
    propagate_properties(workspace, slice)
    if norm_to_one:
        slice = _norm_to_one(slice)
    return slice


def compute_recoil_line(ws_name, axis, relative_mass=None):
    efixed = get_workspace_handle(ws_name).e_fixed
    x_axis = np.arange(axis.start, axis.end, axis.step)
    if not relative_mass:
        relative_mass = 1
    if is_momentum(axis.units):
        momentum_transfer = x_axis
        line = np.square(momentum_transfer * 1.e10 * constants.hbar) / (2 * relative_mass * constants.neutron_mass) /\
            (constants.elementary_charge / 1000)
    elif is_twotheta(axis.units):
        tth = x_axis * np.pi / 180.
        if 'Direct' in get_workspace_handle(ws_name).e_mode:
            line = efixed * (2 - 2 * np.cos(tth)) / (relative_mass + 1 - np.cos(tth))
        else:
            line = efixed * (2 - 2 * np.cos(tth)) / (relative_mass - 1 + np.cos(tth))
    else:
        raise RuntimeError("units of axis not recognised")
    return x_axis, line


def _norm_to_one(workspace):
    return workspace / np.nanmax(np.abs(workspace.get_signal()))


def is_sliceable(workspace):
    ws = get_workspace_handle(workspace)
    if isinstance(ws, PixelWorkspace):
        return True
    else:
        validator = WorkspaceUnitValidator('DeltaE')
        try:
            isvalid = isinstance(ws, Workspace) and validator.isValid(ws.raw_ws) == ''
        except RuntimeError:
            return False
        return isvalid
