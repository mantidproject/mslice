from __future__ import (absolute_import, division, print_function)

import mslice.plotting.pyplot as plt
import mslice.app as app
import numpy as np
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.cli.helperfunctions import _check_workspace_type, _check_workspace_name, _intensity_to_action, \
    _intensity_to_workspace, _function_to_intensity
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
from mslice.plotting import units
from mantid.plots.helperfunctions import get_normalization, get_md_data1d, get_wksp_index_dist_and_label, \
    get_spectrum, get_md_data2d_bin_bounds, get_data_uneven_flag, get_distribution, get_matrix_2d_data

@plt.set_category(plt.CATEGORY_CUT)
def errorbar(axes, workspace, *args, **kwargs):
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
    label = kwargs.pop('label', None)
    label = workspace.name if label is None else label

    if isinstance(workspace, HistogramWorkspace):
        (normalization, kwargs) = get_normalization(workspace.raw_ws, **kwargs)
        (x, y, dy) = get_md_data1d(workspace.raw_ws, normalization)
        dx = None
    else:
        (wkspIndex, distribution, kwargs) = get_wksp_index_dist_and_label(workspace.raw_ws, **kwargs)
        (x, y, dy, dx) = get_spectrum(workspace.raw_ws, wkspIndex, distribution, withDy=True, withDx=True)
    if 'DeltaE' in x_units:
        x = [units.EnergyTransferUnits(v) for v in x]
    axes.errorbar(x, y, dy, dx, *args, label=label, **kwargs)

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
        cur_canvas.manager.add_cut_plot(presenter, workspace.name.rsplit('_', 1)[0])
    cur_fig.canvas.draw()
    axes.pchanged()  # This call is to let the waterfall callback know to update
    return axes.lines


@plt.set_category(plt.CATEGORY_SLICE)
def pcolormesh(axes, workspace, *args, **kwargs):
    """
    Same as the CLI PlotSlice but returns the relevant axes object.
    """
    from mslice.app.presenters import get_slice_plotter_presenter, cli_slice_plotter_presenter
    _check_workspace_name(workspace)
    workspace = get_workspace_handle(workspace)
    _check_workspace_type(workspace, HistogramWorkspace)

    # slice cache needed from main slice plotter presenter
    if is_gui() and GlobalFigureManager.get_active_figure().plot_handler is not None:
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
        plot_handler = GlobalFigureManager.get_active_figure().plot_handler
        intensity_action = getattr(plot_window, _intensity_to_action[intensity])
        plot_handler.set_intensity(intensity_action)
        intensity_action.setChecked(True)

        # Set intensity properties for generated script to use
        if not is_gui():
            for key, value in _function_to_intensity.items():
                if value == intensity:
                    intensity_method = key
                    break
            plot_handler.intensity = True
            plot_handler.intensity_method = intensity_method
            plot_handler.temp = temperature
            plot_handler.temp_dependent = True if temperature is not None else False
            plot_handler._slice_plotter_presenter._slice_cache[plot_handler.ws_name].colourmap = kwargs.get('cmap')

    if not workspace.is_PSD and not slice_cache.rotated:
        workspace = Transpose(OutputWorkspace=workspace.name, InputWorkspace=workspace, store=False)

    x_axis = slice_cache.energy_axis if slice_cache.rotated else slice_cache.momentum_axis
    y_axis = slice_cache.momentum_axis if slice_cache.rotated else slice_cache.energy_axis
    if isinstance(workspace, HistogramWorkspace):
        (normalization, kwargs) = get_normalization(workspace.raw_ws, **kwargs)
        x, y, z = get_md_data2d_bin_bounds(workspace.raw_ws, normalization)
    else:
        (aligned, kwargs) = get_data_uneven_flag(workspace.raw_ws, **kwargs)
        (distribution, kwargs) = get_distribution(workspace.raw_ws, **kwargs)
        if aligned:
            kwargs['pcolortype'] = 'mesh'
            return plotfunctions._pcolorpieces(axes, workspace.raw_ws, distribution, *args, **kwargs)
        else:
            (x, y, z) = get_matrix_2d_data(workspace.raw_ws, distribution, histogram2D=True)
    if 'DeltaE' in x_axis.units:
        x = x.astype(np.dtype(units.EnergyTransferUnits))
    if 'DeltaE' in y_axis.units:
        y = y.astype(np.dtype(units.EnergyTransferUnits))
    axes.pcolormesh(x, y, z, *args, **kwargs)

    axes.set_title(workspace.name[2:], picker=SLICE_PICKER_TOL_PTS)
    comment = get_comment(str(workspace.name))
    axes.get_xaxis().set_units(x_axis.units)
    axes.get_yaxis().set_units(y_axis.units)
    # labels
    axes.set_xlabel(get_display_name(x_axis.units, comment), picker=SLICE_PICKER_TOL_PTS)
    axes.set_ylabel(get_display_name(y_axis.units, comment), picker=SLICE_PICKER_TOL_PTS)
    axes.set_xlim(x_axis.start, x_axis.end)
    axes.set_ylim(y_axis.start, y_axis.end)
    return axes.collections[0]  # Quadmesh object
