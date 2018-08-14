from __future__ import (absolute_import, division, print_function)
import mslice.plotting.pyplot as plt
from mslice.models.cut.cut_plotter import CutPlotter
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from ..labels import get_display_name, generate_legend, CUT_INTENSITY_LABEL


PICKER_TOL_PTS = 3


def draw_interactive_cut(workspace):
    cur_fig = plt.gcf()
    cur_canvas = cur_fig.canvas
    ax = plt.gca()
    if not cur_canvas.manager.has_plot_handler():
        cur_canvas.restore_region(cur_canvas.manager.get_cut_background())
        self._create_cut(workspace)
    try:
        children = cur_fig.get_children()
        for artist in children:
            ax.draw_artist(artist)
        cur_canvas.blit(ax.clipbox)
    except AttributeError:
        cur_canvas.draw_idle()
    plt.show()


@plt.set_category(plt.CATEGORY_CUT)
def plot_cut_impl(workspace, x_units, intensity_range=None, plot_over=False, legend=None):
    legend = workspace.name if legend is None else legend
    if not plot_over:
        plt.cla()
    cur_fig = plt.gcf()
    cur_canvas = cur_fig.canvas
    ax = cur_fig.add_subplot(111, projection='mantid')
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
        cur_canvas.manager.add_cut_plot(None, workspace)
    cur_fig.canvas.draw()


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
