"""Defines the additional mslice commands on top of the standard matplotlib plotting commands
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Mantid Tools imported for convenience
from __future__ import (absolute_import, division, print_function)

import os.path as ospath

from mslice.app import MAIN_WINDOW

from mslice.workspace.base import WorkspaceBase as Workspace

# Helper tools
from mslice.models.workspacemanager.workspace_provider import workspace_exists
from mslice.presenters.slice_plotter_presenter import Axis as _Axis
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def _process_axis(axis, fallback_index, input_workspace):
    # if axis is None:
    #     axis = _SLICE_ALGORITHM.get_available_axis(input_workspace)[fallback_index]

    # check to see if axis is just a name e.g 'DeltaE' or a full binning spec e.g. 'DeltaE,0,1,100'
    if ',' in axis:
        axis = _string_to_axis(axis)
    else:
        axis = _Axis(units=axis, start=None, end=None, step=None) # The model will fill in the rest
    return axis

def _string_to_axis(string):
    axis = string.split(',')
    if len(axis) != 4:
        raise ValueError('axis should be specified in format <name>,<start>,<end>,<step_size>')
    name = axis[0].strip()
    try:
        start = float(axis[1])
    except ValueError:
        raise ValueError("start '%s' is not a valid float"%axis[1])
    try:
        end = float(axis[2])
    except ValueError:
        raise ValueError("end '%s' is not a valid float"%axis[2])

    try:
        step = float(axis[3])
    except ValueError:
        raise ValueError("step '%s' is not a valid float"%axis[3])
    return _Axis(name, start, end, step)


# -----------------------------------------------------------------------------
# Command functions
# -----------------------------------------------------------------------------

def load(path):
    """ Load a workspace from a file.

    Keyword Arguments:
        path -- full path to input file (string)
    """
    if not isinstance(path, str):
        raise RuntimeError('path given to load must be a string')
    if not ospath.exists(path):
        raise RuntimeError('could not find the path %s' % path)
    MAIN_WINDOW.dataloader_presenter.load_workspace([path])
    return get_workspace_handle(ospath.splitext(ospath.basename(path))[0])


def MakeProjection(InputWorkspace, Axis1, Axis2, Units='meV'):
    """ Calculate projections of workspace.

       Keyword Arguments:
           InputWorkspace -- Workspace to project, can be either python handle to workspace or a string containing the
           workspace name.
           Axis1 -- The first axis of projection (string)
           Axis2 -- The second axis of the projection (string)
           Units -- The energy units (string) [default: 'meV']

       """
    if isinstance(InputWorkspace, Workspace):
        InputWorkspace = InputWorkspace.name
    if not isinstance(InputWorkspace, str):
        raise TypeError('InputWorkspace must be a workspace or a workspace name')
    if not workspace_exists(InputWorkspace):
        raise TypeError('InputWorkspace %s could not be found.' % InputWorkspace)

    proj_ws = MAIN_WINDOW.powder_presenter.calc_projection(InputWorkspace, Axis1, Axis2, Units)
    MAIN_WINDOW.powder_presenter.after_projection([proj_ws])
    return proj_ws
