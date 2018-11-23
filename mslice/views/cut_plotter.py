from __future__ import (absolute_import, division, print_function)
import mslice.plotting.pyplot as plt
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from mslice.models.labels import get_display_name, CUT_INTENSITY_LABEL
from mslice.plotting.globalfiguremanager import GlobalFigureManager

PICKER_TOL_PTS = 3


def draw_interactive_cut(workspace):
    cur_fig = plt.gcf()
    cur_canvas = cur_fig.canvas
    ax = plt.gca()

    # disconnect picking in interactive cut
    cur_canvas.manager.picking_connected(False)
    cur_canvas.manager.button_pressed_connected(False)

    if not cur_canvas.manager.has_plot_handler():
        cur_canvas.restore_region(cur_canvas.manager.get_cut_background())
        _create_cut(workspace)
    try:
        children = cur_fig.get_children()
        for artist in children:
            ax.draw_artist(artist)
        cur_canvas.blit(ax.clipbox)
    except AttributeError:
        cur_canvas.draw_idle()
    plt.show()


@plt.set_category(plt.CATEGORY_CUT)
def plot_cut_impl(workspace, presenter, x_units, intensity_range=None, plot_over=False, legend=None, is_gui=True):
    legend = workspace.name if legend is None else legend
    if not plot_over and is_gui:
        plt.cla()

    cur_fig = plt.gcf()
    cur_canvas = cur_fig.canvas
    ax = cur_fig.add_subplot(1, 1, 1, projection='mantid')

    legend = workspace.name if legend is None else legend
    ax.errorbar(workspace.raw_ws, 'o-', label=legend, picker=PICKER_TOL_PTS)
    ax.set_ylim(*intensity_range) if intensity_range is not None else ax.autoscale()
    if cur_canvas.manager.window.action_toggle_legends.isChecked():
        leg = ax.legend(fontsize='medium')
        leg.draggable()
    ax.set_xlabel(get_display_name(x_units, get_comment(workspace)), picker=PICKER_TOL_PTS)
    ax.set_ylabel(CUT_INTENSITY_LABEL, picker=PICKER_TOL_PTS)
    if not plot_over:
        cur_canvas.set_window_title(workspace.name)
        cur_canvas.manager.update_grid()
    if not cur_canvas.manager.has_plot_handler():
        cur_canvas.manager.add_cut_plot(presenter, workspace.name.rsplit('_', 1)[0])
    cur_fig.canvas.draw()
    return ax.lines


def _create_cut():
    canvas = plt.gcf().canvas
    # don't include axis ticks in the saved background
    canvas.figure.gca().xaxis.set_visible(False)
    canvas.figure.gca().yaxis.set_visible(False)
    canvas.draw()
    canvas.manager.set_cut_background(canvas.copy_from_bbox(plt.gcf().canvas.figure.bbox))

    canvas.figure.gca().xaxis.set_visible(True)
    canvas.figure.gca().yaxis.set_visible(True)
    canvas.draw()


def cut_figure_exists():
    return GlobalFigureManager.active_cut_figure_exists()
