from __future__ import (absolute_import, division, print_function)

import mantid.kernel

from ._mslice_commands import app, DEFAULT_CMAP, get_workspace_handle
from mslice.plotting.cli_helperfunctions import is_slice, is_cut
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.plotting.cli_helperfunctions import _check_workspace_type, _check_workspace_name
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.plotting.plot_window.slice_plot import SlicePlot

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


def ConvertToChi(figure_number):
    """
    Converts to the Dynamical susceptibility Chi''(Q,E) on Slice Plot
    :param figure_number: The slice plot figure number returned when the plot was made.
    :return:
    """
    global figure_manager

    plot_handler = figure_manager.get_figure_by_number(figure_number)._plot_handler
    if isinstance(plot_handler, SlicePlot):
        plot_handler.plot_window.action_chi_qe.trigger()
    else:
        print('This function cannot be used on a Cut')


def ConvertToChiMag(figure_number):
    """
        Converts to the magnetic dynamical susceptibility Chi''(Q,E magnetic on Slice Plot
        :param figure_number: The slice plot figure number returned when the plot was made.
        :return:
        """
    global figure_manager

    plot_handler = figure_manager.get_figure_by_number(figure_number)._plot_handler
    if isinstance(plot_handler, SlicePlot):
        plot_handler.plot_window.action_chi_qe_magnetic.trigger()
    else:
        print('This function cannot be used on a Cut')


def ConvertToCrossSection(figure_number):
    """
        Converts to the double differential cross-section d2sigma/dOmega.dE  on Slice Plot
        :param figure_number: The slice plot figure number returned when the plot was made.
        :return:
        """
    global figure_manager

    plot_handler = figure_manager.get_figure_by_number(figure_number)._plot_handler
    if isinstance(plot_handler, SlicePlot):
        plot_handler.plot_window.action_d2sig_dw_de.trigger()
    else:
        print('This function cannot be used on a Cut')


def SymmetriseSQE(figure_number):
    """
        Converts to the double differential cross-section d2sigma/dOmega.dE  on Slice Plot
        :param figure_number: The slice plot figure number returned when the plot was made.
        :return:
        """
    global figure_manager

    plot_handler = figure_manager.get_figure_by_number(figure_number)._plot_handler
    if isinstance(plot_handler, SlicePlot):
        plot_handler.plot_window.action_symmetrised_sqe.trigger()
    else:
        print('This function cannot be used on a Cut')


def ConvertToGDOS(figure_number):
    """
        Converts to symmetrised S(Q,E) (w.r.t. energy using temperature Boltzmann factor) on Slice Plot
        :param figure_number: The slice plot figure number returned when the plot was made.
        :return:
        """
    global figure_manager

    plot_handler = figure_manager.get_figure_by_number(figure_number)._plot_handler
    if isinstance(plot_handler, SlicePlot):
        plot_handler.plot_window.action_gdos.trigger()
    else:
        print('This function cannot be used on a Cut')


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
