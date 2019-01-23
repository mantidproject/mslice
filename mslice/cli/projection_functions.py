from __future__ import (absolute_import, division, print_function)

import mslice.app as app
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.cli.helperfunctions import _check_workspace_type, _check_workspace_name
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.app.presenters import cli_cut_plotter_presenter, cli_slice_plotter_presenter, is_gui
from mslice.util.mantid.mantid_algorithms import Transpose
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from mslice.models.labels import get_display_name
from mantid.plots import plotfunctions
from mslice.app.presenters import get_slice_plotter_presenter
from mslice.views.slice_plotter import create_slice_figure

PICKER_TOL_PTS = 5


def PlotCutMsliceProjection(InputWorkspace, IntensityStart=0, IntensityEnd=0, PlotOver=False):
    """
    Same as the cli PlotCut but returns the relevant axes object.
    """

    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")
    if IntensityStart == 0 and IntensityEnd == 0:
        intensity_range = None
    else:
        intensity_range = (IntensityStart, IntensityEnd)
    lines = cli_cut_plotter_presenter.plot_cut_from_workspace(workspace, intensity_range=intensity_range,
                                                              plot_over=PlotOver)

    return lines


def PlotSliceMsliceProjection(axes, workspace, *args, **kwargs):
    """
    Same as the CLI PlotSlice but returns the relevant axes object.
    """
    _check_workspace_name(workspace)
    workspace = get_workspace_handle(workspace)
    _check_workspace_type(workspace, HistogramWorkspace)

    # slice cache needed from main slice plotter presenter
    if is_gui():
        cli_slice_plotter_presenter._slice_cache = app.MAIN_WINDOW.slice_plotter_presenter._slice_cache
    else:
        create_slice_figure(workspace.name[2:], get_slice_plotter_presenter())

    slice_cache = get_slice_plotter_presenter().get_slice_cache(workspace)

    if not workspace.is_PSD and not slice_cache.rotated:
        workspace = Transpose(OutputWorkspace=workspace.name, InputWorkspace=workspace, store=False)
    image = plotfunctions.pcolormesh(axes, workspace.raw_ws, *args, **kwargs)
    axes.set_title(workspace.name[2:], picker=PICKER_TOL_PTS)
    x_axis = slice_cache.energy_axis if slice_cache.rotated else slice_cache.momentum_axis
    y_axis = slice_cache.momentum_axis if slice_cache.rotated else slice_cache.energy_axis
    comment = get_comment(str(workspace.name))
    axes.get_xaxis().set_units(x_axis.units)
    axes.get_yaxis().set_units(y_axis.units)
    # labels
    axes.set_xlabel(get_display_name(x_axis.units, comment), picker=PICKER_TOL_PTS)
    axes.set_ylabel(get_display_name(y_axis.units, comment), picker=PICKER_TOL_PTS)
    axes.set_xlim(x_axis.start, x_axis.end)
    axes.set_ylim(y_axis.start, y_axis.end)

    return image
