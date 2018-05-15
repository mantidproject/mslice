"""Defines the additional mslice commands on top of the standard matplotlib plotting commands
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Mantid Tools imported for convenience
from __future__ import (absolute_import, division, print_function)

import os.path as ospath
from mantid.api import IMDWorkspace as _IMDWorkspace

from mslice.app import MAIN_WINDOW
from mslice.util.mantid import run_algorithm
# Helper tools
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, workspace_exists
from mslice.models.alg_workspace_ops import get_axis_range, get_available_axis
from mslice.models.axis import Axis
from mslice.workspace.base import WorkspaceBase as Workspace
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

_POWDER_PROJECTION_MODEL = _MantidProjectionCalculator()
_SLICE_ALGORITHM = _MantidSliceAlgorithm()
_SLICE_MODEL = _MatplotlibSlicePlotter(_SLICE_ALGORITHM)
_CUT_ALGORITHM = _MantidCutAlgorithm()
_CUT_PLOTTER = MatplotlibCutPlotter(_CUT_ALGORITHM)

# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------


def _string_to_axis(string):
    axis = string.split(',')
    if len(axis) != 4:
        raise ValueError('axis should be specified in format <name>,<start>,<end>,<step_size>')
    return Axis(axis[0], axis[1], axis[2], axis[3])


def _string_to_integration_axis(string):
    '''Allows step to be omitted and set to default value'''
    axis_str = string.split(',')
    if len(axis_str) < 3:
        raise ValueError('axis should be specified in format <name>,<start>,<end>')
    valid_axis = Axis(axis_str[0], axis_str[1], axis_str[2], 0)
    try:
        valid_axis.step = axis_str[3]
    except IndexError:
        valid_axis.step = valid_axis.end - valid_axis.start
    return valid_axis

def _process_axis(axis, fallback_index, input_workspace, string_function=_string_to_axis):
    available_axes = get_available_axis(input_workspace)
    if axis is None:
        axis = available_axes[fallback_index]
    # check to see if axis is just a name e.g 'DeltaE' or a full binning spec e.g. 'DeltaE,0,1,100'
    if ',' in axis:
        axis = string_function(axis)
    elif axis in available_axes:
        range = get_axis_range(input_workspace, axis)
        range = map(float, range)
        axis = Axis(units=axis, start=range[0], end=range[1], step=range[2])
    else:
        raise RuntimeError("Axis '%s' not recognised. Workspace has these axes: %s " %
                           (axis, ', '.join(available_axes)))
    return axis





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


def Slice(InputWorkspace, Axis1=None, Axis2=None, NormToOne=False):
    """ Slices workspace.

       Keyword Arguments:
       InputWorkspace -- The workspace to slice. The parameter can be either a python handle to the workspace
       OR the workspace name as a string.

       Axis1 -- The x axis of the slice. If not specified will default to |Q| (or Degrees).
       Axis2 -- The y axis of the slice. If not specified will default to DeltaE
       Axis Format:-
           Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
           or just the name e.g. 'DeltaE'. In that case, the start and end will default to the range in the data.

       NormToOne -- if true the slice will be normalized to one.

       """

    workspace = get_workspace_handle(InputWorkspace)
    x_axis = _process_axis(Axis1, 0, workspace)
    y_axis = _process_axis(Axis2, 1 if workspace.is_PSD else 2, workspace)

    return run_algorithm('Slice', InputWorkspace=workspace, XAxis=x_axis.to_dict(), YAxis=y_axis.to_dict(), EMode=workspace.e_mode,
                         PSD=workspace.is_PSD, NormToOne = NormToOne)

def Cut(InputWorkspace, CutAxis=None, IntegrationAxis=None, NormToOne=False):
    """ Cuts workspace.

     Keyword Arguments:
    InputWorkspace -- Workspace to cut. The parameter can be either a python handle to the workspace
    OR the workspace name as a string.

    CutAxis -- The x axis of the cut. If not specified will default to |Q| (or Degrees).
    IntegrationAxis --  The integration axis of the cut. If not specified will default to DeltaE.
    Axis Format:-
           Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5' (step_size may be
           omitted for the integration axis) or just the name e.g. 'DeltaE'. In that case, the start and end will
           default to the range in the data.

    NormToOne -- if true the cut will be normalized to one.

    """
    workspace = get_workspace_handle(InputWorkspace)
    cut_axis = _process_axis(CutAxis, 0, workspace)
    integration_axis = _process_axis(IntegrationAxis, 1 if workspace.is_PSD else 2,
                                     workspace, string_function=_string_to_integration_axis)

    return run_algorithm('Cut', InputWorkspace=workspace, CutAxis=cut_axis.to_dict(), IntegrationAxis=integration_axis.to_dict(),
                         EMode=workspace.e_mode, PSD=workspace.is_PSD, NormToOne=NormToOne)


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

    input_workspace = get_workspace_handle(input_workspace)
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
    if isinstance(input_workspace, Workspace):
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
    if isinstance(input_workspace, Workspace):
        input_workspace = input_workspace.getName()
    cut_axis = _process_axis(cut_axis, None, input_workspace)
    _CUT_PLOTTER.plot_cut(input_workspace, cut_axis, integration_start, integration_end, normalize, intensity_start,
                          intensity_end, plot_over=hold)
