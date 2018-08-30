"""Defines the additional mslice commands on top of the standard matplotlib plotting commands
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Mantid Tools imported for convenience
from __future__ import (absolute_import, division, print_function)

import os.path as ospath

import mslice.app as app

from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, workspace_exists
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.alg_workspace_ops import get_axis_range, get_available_axes
from mslice.models.axis import Axis
from mslice.models.cmap import DEFAULT_CMAP
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.workspace.base import WorkspaceBase as Workspace
from mslice.workspace.workspace import Workspace as MatrixWorkspace
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.histogram_workspace import HistogramWorkspace


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
    available_axes = get_available_axes(input_workspace)
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


def _check_workspace_name(workspace):
    if isinstance(workspace, Workspace):
        return
    if not isinstance(workspace, str):
        raise TypeError('InputWorkspace must be a workspace or a workspace name')
    if not workspace_exists(workspace):
        raise TypeError('InputWorkspace %s could not be found.' % workspace)

def _check_workspace_type(workspace):
    if workspace.is_PSD:
        if isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type - run MakeProjection first.")
        if not isinstance(workspace, PixelWorkspace):
            raise RuntimeError("Incorrect workspace type.")
    else:
        if not isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type.")

# -----------------------------------------------------------------------------
# Command functions
# -----------------------------------------------------------------------------

def Load(path):
    """ Load a workspace from a file.

    Keyword Arguments:
        path -- full path to input file (string)
    """
    if not isinstance(path, str):
        raise RuntimeError('path given to load must be a string')
    if not ospath.exists(path):
        raise RuntimeError('could not find the path %s' % path)
    app.MAIN_WINDOW.dataloader_presenter.load_workspace([path])
    return get_workspace_handle(ospath.splitext(ospath.basename(path))[0])


def MakeProjection(InputWorkspace, Axis1, Axis2, Units='meV'):
    """ Calculate projections of workspace.

       Keyword Arguments:
           InputWorkspace -- Workspace to project, can be either python handle
            to the workspace or a string containing the workspace name.
           Axis1 -- The first axis of projection (string)
           Axis2 -- The second axis of the projection (string)
           Units -- The energy units (string) [default: 'meV']

       """
    _check_workspace_name(InputWorkspace)

    proj_ws = app.MAIN_WINDOW.powder_presenter.calc_projection(InputWorkspace, Axis1, Axis2, Units)
    app.MAIN_WINDOW.powder_presenter.after_projection([proj_ws])
    return proj_ws


def Slice(InputWorkspace, Axis1=None, Axis2=None, NormToOne=False):
    """ Slices workspace.

       Keyword Arguments:
       InputWorkspace -- The workspace to slice. The parameter can be either a python handle to the workspace
       OR the workspace name as a string.

       Axis1 -- The x axis of the slice. If not specified will default to |Q| (or Degrees).
       Axis2 -- The y axis of the slice. If not specified will default to DeltaE
       Axis Format:-
            Either a string in format '<name>, <start>, <end>, <step_size>' e.g.
            'DeltaE,0,100,5'  or just the name e.g. 'DeltaE'. In that case, the
            start and end will default to the range in the data.

       NormToOne -- if true the slice will be normalized to one.

       """
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace)
    x_axis = _process_axis(Axis1, 0, workspace)
    y_axis = _process_axis(Axis2, 1 if workspace.is_PSD else 2, workspace)
    return app.MAIN_WINDOW.slice_plotter_presenter.create_slice(workspace, x_axis, y_axis, None, None, NormToOne,
                                                                DEFAULT_CMAP)


def Cut(InputWorkspace, CutAxis=None, IntegrationAxis=None, NormToOne=False):
    """ Cuts workspace.
     Keyword Arguments:
    InputWorkspace -- Workspace to cut. The parameter can be either a python
                      handle to the workspace OR the workspace name as a string.

    CutAxis -- The x axis of the cut. If not specified will default to |Q| (or Degrees).
    IntegrationAxis --  The integration axis of the cut. If not specified will default to DeltaE.
    Axis Format:-
            Either a string in format '<name>, <start>, <end>, <step_size>' e.g.
            'DeltaE,0,100,5' (step_size may be omitted for the integration axis)
            or just the name e.g. 'DeltaE'. In that case, the start and end will
            default to the full range of the data.

    NormToOne -- if true the cut will be normalized to one.

    """
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace)
    cut_axis = _process_axis(CutAxis, 0, workspace)
    integration_axis = _process_axis(IntegrationAxis, 1 if workspace.is_PSD else 2,
                                     workspace, string_function=_string_to_integration_axis)
    cut = compute_cut(workspace, cut_axis, integration_axis, NormToOne, store=True)
    app.MAIN_WINDOW.cut_plotter_presenter.update_main_window()
    return cut

def PlotSlice(InputWorkspace, IntensityStart="", IntensityEnd="", Colormap=DEFAULT_CMAP):
    """
    Creates mslice standard matplotlib plot of a slice workspace.
    Keyword Arguments:
    InputWorkspace -- Workspace to plot. The parameter can be either a python
                      handle to the workspace OR the workspace name as a string.
    IntensityStart -- Lower bound of the intensity axis (colorbar)
    IntensityEnd -- Upper bound of the intensity axis (colorbar)
    Colormap -- Colormap name as a string. Default is 'viridis'.
    """
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")
    slice_presenter = app.MAIN_WINDOW.slice_plotter_presenter
    slice_presenter.change_intensity(workspace.name, IntensityStart, IntensityEnd)
    slice_presenter.change_colourmap(workspace.name, Colormap)
    slice_presenter.plot_from_cache(workspace)

def PlotCut(InputWorkspace, IntensityStart=0, IntensityEnd=0, PlotOver=False):
    """
    Create mslice standard matplotlib plot of a cut workspace.
    Keyword Arguments:
    InputWorkspace -- Workspace to cut. The parameter can be either a python
                      handle to the workspace OR the workspace name as a string.

    IntensityStart -- Lower bound of the y axis
    IntensityEnd -- Upper bound of the y axis
    PlotOver -- if true the cut will be plotted on an existing figure.
    """
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")
    if IntensityStart == 0 and IntensityEnd == 0:
        intensity_range = None
    else:
        intensity_range = (IntensityStart, IntensityEnd)
    app.MAIN_WINDOW.cut_plotter_presenter.plot_cut_from_workspace(workspace, intensity_range=intensity_range,
                                                                  plot_over=PlotOver)
