from __future__ import (absolute_import, division, print_function)
import mslice.plotting.pyplot as plt
from mslice.models.cut.cut_plotter import CutPlotter
from mslice.models.cut.cut_functions import output_workspace_name, compute_cut_xye
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from ..labels import get_display_name, generate_legend, CUT_INTENSITY_LABEL


PICKER_TOL_PTS = 3


class MatplotlibCutPlotter(CutPlotter):
    def __init__(self):
        self.icut = None

    def plot_cut(self, selected_workspace, cut_axis, integration_axis, norm_to_one, intensity_start,
                 intensity_end, plot_over):
        x, y, e = compute_cut_xye(selected_workspace, cut_axis, integration_axis, norm_to_one)
        output_ws_name = output_workspace_name(selected_workspace, integration_axis.start, integration_axis.end)
        legend = generate_legend(selected_workspace, integration_axis.units, integration_axis.start,
                                 integration_axis.end)
        self.plot_cut_from_xye(x, y, e, cut_axis.units, selected_workspace, (intensity_start, intensity_end),
                               plot_over, output_ws_name, legend)
        plt.show()

    @plt.set_category(plt.CATEGORY_CUT)
    def plot_cut_from_xye(self, x, y, e, x_units, selected_workspace, intensity_range=None, plot_over=False,
                          cut_ws_name=None, legend=None):
        legend = selected_workspace if legend is None else legend
        if not plot_over:
            plt.cla()
        plt.errorbar(x, y, yerr=e, label=legend, marker='o', picker=PICKER_TOL_PTS)
        cur_fig = plt.gcf()
        cur_axes = cur_fig.gca()
        cur_axes.set_ylim(*intensity_range) if intensity_range is not None else cur_axes.autoscale()
        leg = cur_axes.legend(fontsize='medium')
        leg.draggable()
        cur_axes.set_xlabel(get_display_name(x_units, get_comment(selected_workspace)), picker=PICKER_TOL_PTS)
        cur_axes.set_ylabel(CUT_INTENSITY_LABEL, picker=PICKER_TOL_PTS)
        cur_canvas = cur_fig.canvas
        if not plot_over:
            cur_canvas.set_window_title('Cut: ' + selected_workspace)
            cur_canvas.manager.update_grid()
        if not cur_canvas.manager.has_plot_handler():
            self._create_cut(cut_ws_name if cut_ws_name is not None else selected_workspace)
            cur_canvas.restore_region(cur_canvas.manager.get_cut_background())
        try:
            children = cur_fig.get_children()
            for artist in children:
                cur_axes.draw_artist(artist)
            cur_canvas.blit(cur_axes.clipbox)
        except AttributeError:
            cur_canvas.draw_idle()
        plt.show()

    def _create_cut(self, workspace):
        canvas = plt.gcf().canvas
        canvas.manager.add_cut_plot(self, workspace)
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

    def save_cut(self, params):
        #TODO: broken on master, issue #332
        pass
