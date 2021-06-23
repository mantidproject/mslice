from __future__ import (absolute_import, division, print_function)

from mslice.models.labels import get_recoil_label
import mslice.plotting.pyplot as plt

OVERPLOT_COLORS = {1: 'b', 2: 'g', 4: 'r', 'Aluminium': 'g', 'Copper': 'm', 'Niobium': 'y', 'Tantalum': 'b'}
PICKER_TOL_PTS = 5


def plot_cached_slice(slice_workspace, slice_cache):
    _show_plot(slice_workspace, slice_cache)


@plt.set_category(plt.CATEGORY_SLICE)
def create_slice_figure(workspace_name, presenter):
    fig_canvas = plt.gcf().canvas
    fig_canvas.set_window_title(workspace_name)
    fig_canvas.manager.add_slice_plot(presenter, workspace_name)
    fig_canvas.manager.update_grid()
    plt.draw_all()


@plt.set_category(plt.CATEGORY_SLICE)
def _show_plot(slice_cache, workspace):
    cur_fig = plt.gcf()
    cur_fig.clf()
    ax = cur_fig.add_subplot(111, projection='mslice')
    image = ax.pcolormesh(workspace, cmap=slice_cache.colourmap, norm=slice_cache.norm)

    cb = plt.colorbar(image, ax=ax)
    cb.set_label('Intensity (arb. units)', labelpad=20, rotation=270, picker=PICKER_TOL_PTS)

    plt_handler = cur_fig.canvas.manager.plot_handler

    # Because the axis is cleared, RectangleSelector needs to use the new axis
    # otherwise it can't be used after doing an intensity plot (as it clears the axes)
    if plt_handler.icut is not None:
        plt_handler.icut.rect.ax = ax

    plt_handler._update_lines()

    cur_fig.canvas.draw_idle()
    cur_fig.show()

    # This ensures that another slice plotted in the same window saves the plot options
    # as the plot window's showEvent is called only once. The equivalent command is left
    # in the showEvent for use by the CLI.
    if plt_handler.default_options is None:
        plt_handler.save_default_options()


def set_colorbar_label(label):
    plt.gcf().get_axes()[1].set_ylabel(label, rotation=270, labelpad=20)


def remove_line(line):
    plt.gca().lines.remove(line)


def plot_overplot_line(x, y, key, recoil, cache):
    color = OVERPLOT_COLORS[key] if key in OVERPLOT_COLORS else 'c'
    if recoil:
        return overplot_line(x, y, color, get_recoil_label(key), cache.rotated)
    else:
        return overplot_line(x, y, color, key, cache.rotated)


def overplot_line(x, y, color, label, rotated):
    if rotated:
        return plt.gca().plot(y, x, color=color, label=label, alpha=.7,
                              picker=PICKER_TOL_PTS)[0]
    else:
        return plt.gca().plot(x, y, color=color, label=label, alpha=.7,
                              picker=PICKER_TOL_PTS)[0]
