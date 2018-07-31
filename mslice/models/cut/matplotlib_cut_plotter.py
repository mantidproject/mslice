from __future__ import (absolute_import, division, print_function)
import mslice.plotting.pyplot as plt
from mantid.plots import *
from mslice.models.cut.cut_plotter import CutPlotter
from mslice.models.cut.cut_functions import output_workspace_name, compute_cut
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from ..labels import get_display_name, generate_legend, CUT_INTENSITY_LABEL


PICKER_TOL_PTS = 3


class MatplotlibCutPlotter(CutPlotter):
    def __init__(self):
        self.icut = None

    def plot_cut(self, selected_workspace, cut_axis, integration_axis, norm_to_one, intensity_start,
                 intensity_end, plot_over, store=True):
        workspace = get_workspace_handle(selected_workspace)
        cut = compute_cut(workspace, cut_axis, integration_axis, norm_to_one, store)
        legend = generate_legend(workspace.name, integration_axis.units, integration_axis.start,
                                 integration_axis.end)
        self.plot_cut_impl(cut, cut_axis.units, (intensity_start, intensity_end), plot_over, legend)
        plt.show()

    def plot_interactive_cut(self, selected_workspace, cut_axis, integration_axis, norm_to_one, intensity_start,
                 intensity_end, plot_over, store=True):
        self.plot_cut(selected_workspace, cut_axis, integration_axis, norm_to_one, intensity_start, intensity_end,
                      plot_over, store)
        cur_fig = plt.gcf()
        cur_canvas = cur_fig.canvas
        ax = plt.gca()
        if not cur_canvas.manager.has_plot_handler():
            cur_canvas.restore_region(cur_canvas.manager.get_cut_background())
            self._create_cut(selected_workspace)
        try:
            children = cur_fig.get_children()
            for artist in children:
                ax.draw_artist(artist)
            cur_canvas.blit(ax.clipbox)
        except AttributeError:
            cur_canvas.draw_idle()
        plt.show()

    @plt.set_category(plt.CATEGORY_CUT)
    def plot_cut_impl(self, workspace, x_units, intensity_range=None, plot_over=False, legend=None):
        legend = workspace.name if legend is None else legend
        if not plot_over:
            plt.cla()
        cur_fig = plt.gcf()
        cur_canvas = cur_fig.canvas
        ax = cur_fig.add_subplot(111, projection='mantid')
        ax.errorbar(workspace.raw_ws, 'o-', label=legend, picker=PICKER_TOL_PTS)
        ax.set_ylim(*intensity_range) if intensity_range is not None else cur_axes.autoscale()
        if cur_canvas.manager.window.action_toggle_legends.isChecked():
            leg = ax.legend(fontsize='medium')
            leg.draggable()
        ax.set_xlabel(get_display_name(x_units, get_comment(workspace)), picker=PICKER_TOL_PTS)
        ax.set_ylabel(CUT_INTENSITY_LABEL, picker=PICKER_TOL_PTS)
        if not plot_over:
            cur_canvas.set_window_title(workspace.name)
            cur_canvas.manager.update_grid()
        if not cur_canvas.manager.has_plot_handler():
            cur_canvas.manager.add_cut_plot(self, workspace)

    def _create_cut(self, workspace):
        canvas = plt.gcf().canvas
        # don't include axis ticks in the saved background
        canvas.figure.gca().xaxis.set_visible(False)
        canvas.figure.gca().yaxis.set_visible(False)
        canvas.draw()
        canvas.manager.set_cut_background(canvas.copy_from_bbox(plt.gcf().canvas.figure.bbox))

        canvas.figure.gca().xaxis.set_visible(True)
        canvas.figure.gca().yaxis.set_visible(True)
        canvas.draw()

    def set_icut(self, icut):
        if icut is not None:
            if hasattr(icut, 'plot_cut'):
                plt.gcf().canvas.manager.is_icut(True)
                self.icut = icut
            else:
                plt.gcf().canvas.manager.is_icut(icut)
        else:
            self.icut = None
            plt.gcf().canvas.manager.is_icut(False)

    def get_icut(self):
        return self.icut
