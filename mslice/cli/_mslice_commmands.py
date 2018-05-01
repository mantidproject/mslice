"""Defines the additional mslice commands on top of the standard matplotlib plotting commands
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Mantid Tools imported for convenience
from __future__ import (absolute_import, division, print_function)

import os.path as ospath
from mantid.api import IMDWorkspace as _IMDWorkspace
from mantid.api import Workspace as _Workspace
from mantid.kernel.funcinspect import lhs_info as _lhs_info
from mantid.simpleapi import mtd, ConvertUnits, RenameWorkspace # noqa: F401

from mslice.app import MAIN_WINDOW
# Helper tools
from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider as _MantidWorkspaceProvider
from mslice.presenters.slice_plotter_presenter import Axis as _Axis
# Projections
from mslice.models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator as _MantidProjectionCalculator
# Slicing
from mslice.models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter as _MatplotlibSlicePlotter
from mslice.models.slice.mantid_slice_algorithm import MantidSliceAlgorithm as _MantidSliceAlgorithm
# Cutting
from mslice.models.cut.mantid_cut_algorithm import MantidCutAlgorithm as _MantidCutAlgorithm
from mslice.models.cut.matplotlib_cut_plotter import MatplotlibCutPlotter

# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------

_WORKSPACE_PROVIDER = _MantidWorkspaceProvider()
_POWDER_PROJECTION_MODEL = _MantidProjectionCalculator()
_SLICE_ALGORITHM = _MantidSliceAlgorithm()
_SLICE_MODEL = _MatplotlibSlicePlotter(_SLICE_ALGORITHM)
_CUT_ALGORITHM = _MantidCutAlgorithm()
_CUT_PLOTTER = MatplotlibCutPlotter(_CUT_ALGORITHM)

# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def _process_axis(axis, fallback_index, input_workspace):
    if axis is None:
        axis = _SLICE_ALGORITHM.get_available_axis(input_workspace)[fallback_index]

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


def get_projection(input_workspace, axis1, axis2, units='meV'):
    """ Calculate projections of workspace.

    Keyword Arguments:
        input_workspace -- Workspace to project, can be either python handle to workspace or a string containing the
        workspace name.
        axis1 -- The first axis of projection (string)
        axis2 -- The second axis of the projection (string)
        units -- The energy units (string) [default: 'meV']

    """
    if isinstance(input_workspace, _Workspace):
        input_workspace = input_workspace.getName()
    output_workspace = _POWDER_PROJECTION_MODEL.calculate_projection(input_workspace=input_workspace, axis1=axis1,
                                                                     axis2=axis2, units=units)
    try:
        names = _lhs_info('names')
    except RuntimeError:
        names = [output_workspace.getName()]
    if len(names) > 1:
        raise Exception('Too many left hand side arguments, %s' % str(names))
    RenameWorkspace(InputWorkspace=output_workspace, OutputWorkspace=names[0])
    return output_workspace

def get_slice(input_workspace, x=None, y=None, ret_val='both', normalize=False):
    """ Get Slice from workspace as numpy array.

    Keyword Arguments:
    input_workspace -- The workspace to slice. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)

    x -- The x axis of the slice. If not specified will default to Dimension 0 of the workspace
    y -- The y axis of the slice. If not specified will default to Dimension 1 of the workspace
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.

    ret_val -- a string to specify the return value, if ret_val == 'slice' the function will return a single 2D numpy
    array containing the slice data. if ret_value == 'extents' it will return a list containing the range of the slice
    taken [xmin, xmax, ymin, ymax]. if ret_val == 'both' then it will return a tuple (<slice>, <extents>)

    normalize -- if set to True the slice will be normalize to one.

    """
    input_workspace = _WORKSPACE_PROVIDER.get_workspace_handle(input_workspace)
    assert isinstance(input_workspace, _IMDWorkspace)
    x_axis = _process_axis(x, 0, input_workspace)
    y_axis = _process_axis(y, 1, input_workspace)

    slice_array, extents = _SLICE_ALGORITHM.compute_slice(selected_workspace=input_workspace, x_axis=x_axis,
                                                          smoothing=None, y_axis=y_axis, norm_to_one=normalize)
    if ret_val == 'slice':
        return slice_array
    elif ret_val == 'extents':
        return extents
    elif ret_val == 'both':
        return slice_array, extents
    else:
        raise ValueError("ret_val should be 'slice', 'extents' or 'both' and not '%s' " % ret_val)


def plot_slice(input_workspace, x=None, y=None, colormap='viridis', intensity_min=None, intensity_max=None,
               normalize=False):
    """ Plot slice from workspace

    Keyword Arguments:
    input_workspace -- The workspace to slice. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)

    x -- The x axis of the slice. If not specified will default to Dimension 0 of the workspace
    y -- The y axis of the slice. If not specified will default to Dimension 1 of the workspace
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.

    colormap -- a matplotlib colormap.
    intensity_min -- minimum value for intensity
    intensity_max -- maximum value for intensity

    normalize -- if set to True the slice will be normalize to one.

    """

    input_workspace = _WORKSPACE_PROVIDER.get_workspace_handle(input_workspace)
    assert isinstance(input_workspace, _IMDWorkspace)

    x_axis = _process_axis(x, 0, input_workspace)
    y_axis = _process_axis(y, 1, input_workspace)

    _SLICE_MODEL.plot_slice(selected_ws=input_workspace, x_axis=x_axis, y_axis=y_axis, colourmap=colormap,
                            intensity_start=intensity_min, intensity_end=intensity_max,
                            smoothing=None, norm_to_one=normalize)


def get_cut_xye(input_workspace, cut_axis, integration_start, integration_end, normalize=False):
    """Return x, y and e of a cut of a workspace. The function will return a tuple of 3 single dimensional numpy arrays.

    Keyword Arguments
    input_workspace -- The workspace to cut. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)
    cut_axis -- The axis to cut along.
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.
    integration_start -- value to start integrating from
    integration_end -- The value to end the integration at
    normalize -- will normalize the cut data to one if set to true
    """
    if isinstance(input_workspace, _Workspace):
        input_workspace = input_workspace.getName()
    cut_axis = _process_axis(cut_axis, None, input_workspace)
    x, y, e = _CUT_ALGORITHM.compute_cut_xye(input_workspace, cut_axis, integration_start, integration_end,
                                             is_norm=normalize)
    x, y, e = x.squeeze(), y.squeeze(), e.squeeze()
    return x, y, e


def plot_cut(input_workspace, cut_axis, integration_start, integration_end, intensity_start=None,
             intensity_end=None, normalize=False, hold=False):
    """Take a cut of the workspace and plot it.

    Keyword Arguments
    input_workspace -- The workspace to cut. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)
    cut_axis -- The axis to cut along.
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.
    integration_start -- value to start integrating from
    integration_end -- The value to end the integration at
    normalize -- will normalize the cut data to one if set to true
    """
    if isinstance(input_workspace, _Workspace):
        input_workspace = input_workspace.getName()
    cut_axis = _process_axis(cut_axis, None, input_workspace)
    _CUT_PLOTTER.plot_cut(input_workspace, cut_axis, integration_start, integration_end, normalize, intensity_start,
                          intensity_end, plot_over=hold)
