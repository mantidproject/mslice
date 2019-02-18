"""Defines the additional mslice commands on top of the standard matplotlib plotting commands"""
from __future__ import (absolute_import, division, print_function)

import os.path as ospath

import matplotlib as mpl
from mslice.models.workspacemanager.workspace_provider import (get_workspace_handle, rename_workspace,  # noqa: F401
    get_visible_workspace_names, get_workspace_name)  # noqa: F401
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.cmap import DEFAULT_CMAP
import mslice.app as app
from mslice.app import is_gui
from mslice.plotting.globalfiguremanager import GlobalFigureManager
from mslice.cli.helperfunctions import (_string_to_integration_axis, _process_axis, _check_workspace_name,
                                        _check_workspace_type, _update_cache)
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.util.qt.qapp import QAppThreadCall, mainloop
from six import string_types
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.app.presenters import cli_slice_plotter_presenter
from mslice.app.presenters import cli_cut_plotter_presenter

# -----------------------------------------------------------------------------
# Command functions
# -----------------------------------------------------------------------------


def Show():
    """
    Show all figures and start the event loop if necessary
    """
    managers = GlobalFigureManager.get_all_fig_managers()
    if not managers:
        return

    for manager in managers:
        manager.show()

    # Hack: determine at runtime whether we are
    # inside ipython in pylab mode.
    from matplotlib import pyplot

    try:
        ipython_pylab = not pyplot.show._needmain
        # IPython versions >= 0.10 tack the _needmain
        # attribute onto pyplot.show, and always set
        # it to False, when in %pylab mode.
        ipython_pylab = ipython_pylab and mpl.get_backend() != 'WebAgg'
        # TODO: The above is a hack to get the WebAgg backend
        # working with ipython's `%pylab` mode until proper
        # integration is implemented.
    except AttributeError:
        ipython_pylab = False

    # Leave the following as a separate step in case we
    # want to control this behavior with an rcParam.
    if ipython_pylab:
        return

    if not mpl.is_interactive() or mpl.get_backend() == 'WebAgg':
        QAppThreadCall(mainloop)()


def Load(Filename, OutputWorkspace=None):

    """
    Load a workspace from a file.

    :param Filename:  full path to input file (string)
    :return:
    """
    from mslice.app.presenters import get_dataloader_presenter

    if not isinstance(Filename, string_types):
        raise RuntimeError('path given to load must be a string')
    if not ospath.exists(Filename):
        raise RuntimeError('could not find the path %s' % Filename)

    get_dataloader_presenter().load_workspace([Filename])
    name = ospath.splitext(ospath.basename(Filename))[0]
    if OutputWorkspace is not None:
        name = rename_workspace(workspace=ospath.splitext(ospath.basename(Filename))[0], new_name=OutputWorkspace).name

    return get_workspace_handle(name)


def GenerateScript(InputWorkspace, filename):
    from mslice.scripting import generate_script
    _check_workspace_name(InputWorkspace)
    workspace_name = get_workspace_handle(InputWorkspace).name[2:]
    plot_handler = GlobalFigureManager.get_active_figure().plot_handler
    generate_script(ws_name=workspace_name, filename=filename, plot_handler=plot_handler)


def MakeProjection(InputWorkspace, Axis1, Axis2, Units='meV', OutputWorkspace=None, Limits=None):
    """
    Calculate projections of workspace

    :param InputWorkspace: Workspace to project, can be either python handle
    to the workspace or a string containing the workspace name.
    :param Axis1: The first axis of projection (string)
    :param Axis2: The second axis of the projection (string)
    :param Units: The energy units (string) [default: 'meV']
    :return:
    """
    from mslice.app.presenters import get_powder_presenter

    _check_workspace_name(InputWorkspace)
    return get_powder_presenter().calc_projection(InputWorkspace, Axis1, Axis2, Units)


def Slice(InputWorkspace, Axis1=None, Axis2=None, NormToOne=False):
    """
    Slices workspace.

    :param InputWorkspace: The workspace to slice. The parameter can be either a python handle to the workspace
       OR the workspace name as a string.
    :param Axis1: The x axis of the slice. If not specified will default to |Q| (or Degrees).
    :param Axis2: The y axis of the slice. If not specified will default to DeltaE
       Axis Format:-
            Either a string in format '<name>, <start>, <end>, <step_size>' e.g.
            'DeltaE,0,100,5'  or just the name e.g. 'DeltaE'. In that case, the
            start and end will default to the range in the data.
    :param NormToOne: if True the slice will be normalized to one.
    :return:
    """
    from mslice.app.presenters import get_slice_plotter_presenter
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace, PixelWorkspace)
    x_axis = _process_axis(Axis1, 0, workspace)
    y_axis = _process_axis(Axis2, 1 if workspace.is_PSD else 2, workspace)

    return get_slice_plotter_presenter().create_slice(workspace, x_axis, y_axis, None, None, NormToOne, DEFAULT_CMAP)


def Cut(InputWorkspace, CutAxis=None, IntegrationAxis=None, NormToOne=False):
    """
    Cuts workspace.
    :param InputWorkspace: Workspace to cut. The parameter can be either a python
                      handle to the workspace OR the workspace name as a string.
    :param CutAxis: The x axis of the cut. If not specified will default to |Q| (or Degrees).
    :param IntegrationAxis: The integration axis of the cut. If not specified will default to DeltaE.
    Axis Format:-
            Either a string in format '<name>, <start>, <end>, <step_size>' e.g.
            'DeltaE,0,100,5' (step_size may be omitted for the integration axis)
            or just the name e.g. 'DeltaE'. In that case, the start and end will
            default to the full range of the data.
    :param NormToOne: if True the cut will be normalized to one.
    :return:
    """
    from mslice.app.presenters import get_cut_plotter_presenter

    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace, PixelWorkspace)
    cut_axis = _process_axis(CutAxis, 0, workspace)
    integration_axis = _process_axis(IntegrationAxis, 1 if workspace.is_PSD else 2,
                                     workspace, string_function=_string_to_integration_axis)
    cut = compute_cut(workspace, cut_axis, integration_axis, NormToOne, store=True)
    get_cut_plotter_presenter().update_main_window()

    # Create the cut for use by the plot window in a generated script
    if not is_gui():
        _update_cache(get_cut_plotter_presenter(), workspace.name, CutAxis, IntegrationAxis, NormToOne)

    return cut


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
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace, HistogramWorkspace)

    # slice cache needed from main slice plotter presenter
    if is_gui():
        cli_slice_plotter_presenter._slice_cache = app.MAIN_WINDOW.slice_plotter_presenter._slice_cache
    cli_slice_plotter_presenter.change_intensity(workspace.name, IntensityStart, IntensityEnd)
    cli_slice_plotter_presenter.change_colourmap(workspace.name, Colormap)
    cli_slice_plotter_presenter.plot_from_cache(workspace)

    return GlobalFigureManager._active_figure


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
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")

    if IntensityStart == 0 and IntensityEnd == 0:
        intensity_range = None
    else:
        intensity_range = (IntensityStart, IntensityEnd)
    cli_cut_plotter_presenter.plot_cut_from_workspace(workspace, intensity_range=intensity_range,
                                                      plot_over=PlotOver)

    return GlobalFigureManager._active_figure


def KeepFigure(figure_number=None):
    GlobalFigureManager.set_figure_as_kept(figure_number)


def MakeCurrent(figure_number=None):
    GlobalFigureManager.set_figure_as_current(figure_number)


def ConvertToChi(figure_number):
    """
    Converts to the Dynamical susceptibility Chi''(Q,E) on Slice Plot
    :param figure_number: The slice plot figure number returned when the plot was made.
    :return:
    """
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    plot_handler = GlobalFigureManager.get_figure_by_number(figure_number).plot_handler
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
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    plot_handler = GlobalFigureManager.get_figure_by_number(figure_number).plot_handler
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
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    plot_handler = GlobalFigureManager.get_figure_by_number(figure_number).plot_handler
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
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    plot_handler = GlobalFigureManager.get_figure_by_number(figure_number).plot_handler
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
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    plot_handler = GlobalFigureManager.get_figure_by_number(figure_number).plot_handler
    if isinstance(plot_handler, SlicePlot):
        plot_handler.plot_window.action_gdos.trigger()
    else:
        print('This function cannot be used on a Cut')
