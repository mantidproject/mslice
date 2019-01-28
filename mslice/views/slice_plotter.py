from __future__ import (absolute_import, division, print_function)

from mslice.models.labels import recoil_labels
import mslice.plotting.pyplot as plt


OVERPLOT_COLORS = {1: 'b', 2: 'g', 4: 'r', 'Aluminium': 'g', 'Copper': 'm', 'Niobium': 'y', 'Tantalum': 'b'}
PICKER_TOL_PTS = 5


def plot_cached_slice(slice_workspace, slice_cache):
    ax = _show_plot(slice_workspace, slice_cache)
    return ax


@plt.set_category(plt.CATEGORY_SLICE)
def create_slice_figure(workspace_name, presenter):
    fig_canvas = plt.gcf().canvas
    fig_canvas.set_window_title(workspace_name)
    plot_handler = fig_canvas.manager.add_slice_plot(presenter, workspace_name)
    fig_canvas.manager.update_grid()
    plt.draw_all()
    return plot_handler


@plt.set_category(plt.CATEGORY_SLICE)
def _show_plot(slice_cache, workspace):
    cur_fig = plt.gcf()
    cur_fig.clf()
    ax = cur_fig.add_subplot(111, projection='mslice')
    image = ax.pcolormesh(workspace, cmap=slice_cache.colourmap, norm=slice_cache.norm)

    cb = plt.colorbar(image, ax=ax)
    cb.set_label('Intensity (arb. units)', labelpad=20, rotation=270, picker=PICKER_TOL_PTS)

    # Because the axis is cleared, RectangleSelector needs to use the new axis
    # otherwise it can't be used after doing an intensity plot (as it clears the axes)
    if cur_fig.canvas.manager._plot_handler.icut is not None:
        cur_fig.canvas.manager._plot_handler.icut.rect.ax = ax

    cur_fig.canvas.draw_idle()
    cur_fig.show()

    return ax


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
