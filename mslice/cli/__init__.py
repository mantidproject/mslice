from __future__ import (absolute_import, division, print_function)

from ._mslice_commands import *  # noqa: F401


import mantid.kernel
from mslice.models.workspacemanager.workspace_provider import workspace_exists
from mslice.plotting.cli_helperfunctions import is_slice, is_cut
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.workspace.base import WorkspaceBase as Workspace
from mslice.workspace.workspace import Workspace as MatrixWorkspace
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter

# Imports for mslice projections
from matplotlib.axes import Axes
from matplotlib.projections import register_projection


# Separate cutplotter for cli
CLI_CUT_PLOTTER_PRESENTER = CutPlotterPresenter()
CLI_SLICE_PLOTTER_PRESENTER = SlicePlotterPresenter()


def _check_workspace_name(workspace):
    if isinstance(workspace, Workspace):
        return
    if not isinstance(workspace, str):
        raise TypeError('InputWorkspace must be a workspace or a workspace name')
    if not workspace_exists(workspace):
        raise TypeError('InputWorkspace %s could not be found.' % workspace)


def _check_workspace_type(workspace, correct_type):
    """Check a PSD workspace is MatrixWorkspace, or non-PSD is the specified type"""
    if workspace.is_PSD:
        if isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type - run MakeProjection first.")
        if not isinstance(workspace, correct_type):
            raise RuntimeError("Incorrect workspace type.")
    else:
        if not isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type.")


def PlotSlice(InputWorkspace, IntensityStart="", IntensityEnd="", Colormap=DEFAULT_CMAP):
    """
    Creates mslice standard matplotlib plot of a slice workspace.

    :param InputWorkspace: Workspace to plot. The parameter can be either a python
    handle to the workspace OR the workspace name as a string.
    :param IntensityStart: Lower bound of the intensity axis (colorbar)
    :param IntensityEnd: Upper bound of the intensity axis (colorbar)
    :param Colormap: Colormap name as a string. Default is 'viridis'.
    :return:
    """
    global CLI_SLICE_PLOTTER_PRESENTER
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace, HistogramWorkspace)
    slice_presenter = CLI_SLICE_PLOTTER_PRESENTER

    # slice cache needed from main slice plotter presenter
    slice_presenter._slice_cache = app.MAIN_WINDOW.slice_plotter_presenter._slice_cache


    slice_presenter.change_intensity(workspace.name, IntensityStart, IntensityEnd)
    slice_presenter.change_colourmap(workspace.name, Colormap)
    slice_presenter.plot_from_cache(workspace)


def PlotCut(InputWorkspace, IntensityStart=0, IntensityEnd=0, PlotOver=False):
    """
    Create mslice standard matplotlib plot of a cut workspace.

    :param InputWorkspace: Workspace to cut. The parameter can be either a python handle to the workspace
    OR the workspace name as a string.
    :param IntensityStart: Lower bound of the y axis
    :param IntensityEnd: Upper bound of the y axis
    :param PlotOver: if true the cut will be plotted on an existing figure.
    :return:
    """
    global CLI_CUT_PLOTTER_PRESENTER
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")
    if IntensityStart == 0 and IntensityEnd == 0:
        intensity_range = None
    else:
        intensity_range = (IntensityStart, IntensityEnd)
    CLI_CUT_PLOTTER_PRESENTER.plot_cut_from_workspace(workspace, intensity_range=intensity_range,
                                                                  plot_over=PlotOver)


class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):

        if is_cut(*args):
            try:
                mantid.kernel.logger.debug('using mantid.plots.plotfunctions')
                PlotCut(*args, **kwargs)
            except Exception as e:
                print('Mslice Projection Error: ' + repr(e))
        else:
            return Axes.plot(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):

        if is_slice(*args):
            try:
                mantid.kernel.logger.debug('using mantid.plots.plotfunctions')
                PlotSlice(*args, **kwargs)
            except Exception as e:
                print('MSlice Projection Error: ' + repr(e))
        else:
            return Axes.pcolormesh(self, *args, **kwargs)


register_projection(MSliceAxes)
