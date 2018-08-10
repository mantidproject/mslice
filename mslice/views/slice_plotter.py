from __future__ import (absolute_import, division, print_function)

from mslice.models.labels import get_display_name, recoil_labels
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from mslice.util.mantid import run_algorithm
import mslice.plotting.pyplot as plt

OVERPLOT_COLORS = {1: 'b', 2: 'g', 4: 'r', 'Aluminium': 'g', 'Copper': 'm', 'Niobium': 'y', 'Tantalum': 'b'}
PICKER_TOL_PTS = 5


def plot_cached_slice(slice_workspace, slice_cache):
    _show_plot(slice_workspace, slice_cache)

@plt.set_category(plt.CATEGORY_SLICE)
def create_slice(workspace_name, presenter):
    fig_canvas = plt.gcf().canvas
    fig_canvas.set_window_title(workspace_name)
    fig_canvas.manager.add_slice_plot(presenter, workspace_name)
    fig_canvas.manager.update_grid()
    plt.draw_all()


@plt.set_category(plt.CATEGORY_SLICE)
def _show_plot(slice_cache, workspace):
    cur_fig = plt.gcf()
    cur_fig.clf()
    ax = cur_fig.add_subplot(111, projection='mantid')
    if not workspace.is_PSD and not slice_cache.rotated:
        workspace = run_algorithm('Transpose', output_name=workspace.name, InputWorkspace=workspace, store=False)
    image = ax.pcolormesh(workspace.raw_ws, cmap=slice_cache.colourmap, norm=slice_cache.norm)
    ax.set_title(workspace.name[2:], picker=PICKER_TOL_PTS)
    x_axis = slice_cache.energy_axis if slice_cache.rotated else slice_cache.momentum_axis
    y_axis = slice_cache.momentum_axis if slice_cache.rotated else slice_cache.energy_axis
    comment = get_comment(str(workspace.name))
    ax.get_xaxis().set_units(x_axis.units)
    ax.get_yaxis().set_units(y_axis.units)
    # labels
    ax.set_xlabel(get_display_name(x_axis.units, comment), picker=PICKER_TOL_PTS)
    ax.set_ylabel(get_display_name(y_axis.units, comment), picker=PICKER_TOL_PTS)
    ax.set_xlim(x_axis.start)
    ax.set_ylim(y_axis.start)
    cb = plt.colorbar(image, ax=ax)
    cb.set_label('Intensity (arb. units)', labelpad=20, rotation=270, picker=PICKER_TOL_PTS)
    cur_fig.canvas.draw_idle()
    cur_fig.show()


def set_colorbar_label(label):
    plt.gcf().get_axes()[1].set_ylabel(label, rotation=270, labelpad=20)


def remove_line(line):
    plt.gca().lines.remove(line)


def plot_overplot_line(x, y, key, recoil, cache):
    color = OVERPLOT_COLORS[key] if key in OVERPLOT_COLORS else 'c'
    if recoil:  # key is relative mass
        label = recoil_labels[key] if key in recoil_labels else 'Relative mass ' + str(key)
    else:  # key is element name
        label = key
    if cache.rotated:
        return plt.gca().plot(y, x, color=color, label=label, alpha=.7,
                              picker=PICKER_TOL_PTS)[0]
    else:
        return plt.gca().plot(x, y, color=color, label=label, alpha=.7,
                              picker=PICKER_TOL_PTS)[0]
