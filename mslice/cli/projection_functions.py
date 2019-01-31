from __future__ import (absolute_import, division, print_function)

import mslice.plotting.pyplot as plt
import mslice.app as app
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.cli.helperfunctions import _check_workspace_type, _check_workspace_name, _intensity_to_action, \
    _intensity_to_workspace
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.app import is_gui
from mslice.util.mantid.mantid_algorithms import Transpose
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from mslice.models.labels import get_display_name, CUT_INTENSITY_LABEL
from mantid.plots import plotfunctions
from mslice.views.slice_plotter import create_slice_figure
from mslice.views.slice_plotter import PICKER_TOL_PTS as SLICE_PICKER_TOL_PTS
from mslice.views.cut_plotter import PICKER_TOL_PTS as CUT_PICKER_TOL_PTS
from mslice.plotting.globalfiguremanager import GlobalFigureManager


@plt.set_category(plt.CATEGORY_CUT)
def PlotCutMsliceProjection(axes, workspace, *args, **kwargs):
    """
    Same as the cli PlotCut but returns the relevant axes object.
    """
    from mslice.app.presenters import get_cut_plotter_presenter
    cur_fig = plt.gcf()
    cur_canvas = cur_fig.canvas

    _check_workspace_name(workspace)
    workspace = get_workspace_handle(workspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")

    presenter = get_cut_plotter_presenter()

    plot_over = kwargs.pop('plot_over', True)
    intensity_range = kwargs.pop('intensity_range', (None, None))
    x_units = kwargs.pop('x_units', 'None')
    legend = kwargs.pop('label', None)
    legend = workspace.name if legend is None else legend

    plotfunctions.errorbar(axes, workspace.raw_ws, label=legend, *args, **kwargs)

    axes.set_ylim(*intensity_range) if intensity_range is not None else axes.autoscale()
    if cur_canvas.manager.window.action_toggle_legends.isChecked():
        leg = axes.legend(fontsize='medium')
        leg.draggable()
    axes.set_xlabel(get_display_name(x_units, get_comment(workspace)), picker=CUT_PICKER_TOL_PTS)
    axes.set_ylabel(CUT_INTENSITY_LABEL, picker=CUT_PICKER_TOL_PTS)
    if not plot_over:
        cur_canvas.set_window_title(workspace.name)
        cur_canvas.manager.update_grid()
    if not cur_canvas.manager.has_plot_handler():
        plot_handler = cur_canvas.manager.add_cut_plot(presenter, workspace.name.rsplit('_', 1)[0])
        plot_handler.save_default_options()
    cur_fig.canvas.draw()

    return axes.lines


@plt.set_category(plt.CATEGORY_SLICE)
def PlotSliceMsliceProjection(axes, workspace, *args, **kwargs):
    """
    Same as the CLI PlotSlice but returns the relevant axes object.
    """
    from mslice.app.presenters import get_slice_plotter_presenter, cli_slice_plotter_presenter
    _check_workspace_name(workspace)
    workspace = get_workspace_handle(workspace)
    _check_workspace_type(workspace, HistogramWorkspace)

    # slice cache needed from main slice plotter presenter
    if is_gui():
        cli_slice_plotter_presenter._slice_cache = app.MAIN_WINDOW.slice_plotter_presenter._slice_cache
    else:
        # Needed so the figure manager knows about the slice plot handler
        create_slice_figure(workspace.name[2:], get_slice_plotter_presenter())

    slice_cache = get_slice_plotter_presenter().get_slice_cache(workspace)

    intensity = kwargs.pop('intensity', None)
    temperature = kwargs.pop('temperature', None)

    if temperature is not None:
        get_slice_plotter_presenter().set_sample_temperature(workspace.name[2:], temperature)

    if intensity is not None and intensity != 's(q,e)':
        workspace = getattr(slice_cache, _intensity_to_workspace[intensity])
        plot_window = GlobalFigureManager.get_active_figure().window
        plot_handler = GlobalFigureManager.get_active_figure()._plot_handler
        intensity_action = getattr(plot_window, _intensity_to_action[intensity])
        plot_handler.set_intensity(intensity_action)
        intensity_action.setChecked(True)

    if not workspace.is_PSD and not slice_cache.rotated:
        workspace = Transpose(OutputWorkspace=workspace.name, InputWorkspace=workspace, store=False)
    image = plotfunctions.pcolormesh(axes, workspace.raw_ws, *args, **kwargs)
    axes.set_title(workspace.name[2:], picker=SLICE_PICKER_TOL_PTS)
    x_axis = slice_cache.energy_axis if slice_cache.rotated else slice_cache.momentum_axis
    y_axis = slice_cache.momentum_axis if slice_cache.rotated else slice_cache.energy_axis
    comment = get_comment(str(workspace.name))
    axes.get_xaxis().set_units(x_axis.units)
    axes.get_yaxis().set_units(y_axis.units)
    # labels
    axes.set_xlabel(get_display_name(x_axis.units, comment), picker=SLICE_PICKER_TOL_PTS)
    axes.set_ylabel(get_display_name(y_axis.units, comment), picker=SLICE_PICKER_TOL_PTS)
    axes.set_xlim(x_axis.start, x_axis.end)
    axes.set_ylim(y_axis.start, y_axis.end)
    return image
