from __future__ import (absolute_import, division, print_function)

import mantid.kernel

from ._mslice_commands import *
from mslice.plotting.cli_helperfunctions import is_slice, is_cut
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.plotting.cli_helperfunctions import _check_workspace_type, _check_workspace_name
from mslice.workspace.histogram_workspace import HistogramWorkspace

# Imports for mslice projections
from matplotlib.axes import Axes
from matplotlib.projections import register_projection


# Separate cutplotter for cli
CLI_CUT_PLOTTER_PRESENTER = CutPlotterPresenter()
CLI_SLICE_PLOTTER_PRESENTER = SlicePlotterPresenter()
current_figure_number = None
figure_manager = None


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
    global current_figure_number
    global figure_manager
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace, HistogramWorkspace)
    slice_presenter = CLI_SLICE_PLOTTER_PRESENTER

    # slice cache needed from main slice plotter presenter
    slice_presenter._slice_cache = app.MAIN_WINDOW.slice_plotter_presenter._slice_cache
    slice_presenter.change_intensity(workspace.name, IntensityStart, IntensityEnd)
    slice_presenter.change_colourmap(workspace.name, Colormap)
    manager = slice_presenter.plot_from_cache(workspace)

    figure_manager = manager._current_figs
    current_figure_number = manager.number
    return manager


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
    global figure_manager
    global current_figure_number
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")
    if IntensityStart == 0 and IntensityEnd == 0:
        intensity_range = None
    else:
        intensity_range = (IntensityStart, IntensityEnd)
    manager = CLI_CUT_PLOTTER_PRESENTER.plot_cut_from_workspace(workspace, intensity_range=intensity_range,
                                                                plot_over=PlotOver)
    figure_manager = manager._current_figs
    current_figure_number = manager.number
    return manager


def KeepFigure(figure_number=None):
    global figure_manager
    global current_figure_number

    if figure_number is not None:
        figure_manager.set_figure_as_kept(figure_number)
    else:
        figure_manager.set_figure_as_kept(current_figure_number)


def MakeCurrent(figure_number=None):
    global figure_manager
    global current_figure_number

    if figure_number is not None:
        figure_manager.set_figure_as_current(figure_number)
    else:
        figure_manager.set_figure_as_current(current_figure_number)


class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):

        if is_cut(*args):
            try:
                mantid.kernel.logger.debug('using mantid.plots.plotfunctions')
                PlotCut(*args, **kwargs)
                return current_figure_number
            except Exception as e:
                print('Mslice Projection Error: ' + repr(e))
        else:
            return Axes.plot(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):

        if is_slice(*args):
            try:
                mantid.kernel.logger.debug('using mantid.plots.plotfunctions')
                PlotSlice(*args, **kwargs)
                return current_figure_number
            except Exception as e:
                print('MSlice Projection Error: ' + repr(e))
        else:
            return Axes.pcolormesh(self, *args, **kwargs)


register_projection(MSliceAxes)
