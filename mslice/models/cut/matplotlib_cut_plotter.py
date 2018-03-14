from __future__ import (absolute_import, division, print_function)
import mslice.plotting.pyplot as plt
from mslice.app import MPL_COMPAT
from .cut_plotter import CutPlotter
from .mantid_cut_algorithm import output_workspace_name

INTENSITY_LABEL = 'Signal/#Events'
picker=3


class MatplotlibCutPlotter(CutPlotter):
    def __init__(self, cut_algorithm):
        self._cut_algorithm = cut_algorithm
        self.workspace_provider = None
        self.background = None
        self.icut = None

    def plot_cut(self, selected_workspace, cut_axis, integration_start, integration_end, norm_to_one, intensity_start,
                 intensity_end, plot_over):
        x, y, e = self._cut_algorithm.compute_cut_xye(selected_workspace, cut_axis, integration_start, integration_end,
                                                      norm_to_one)
        output_ws_name = output_workspace_name(selected_workspace, integration_start, integration_end)
        integrated_dim = self._cut_algorithm.get_other_axis(selected_workspace, cut_axis)
        legend = self._generate_legend(selected_workspace, integrated_dim, integration_start, integration_end)
        self.plot_cut_from_xye(x, y, e, cut_axis.units, selected_workspace, (intensity_start, intensity_end),
                               plot_over, output_ws_name, legend)

    def plot_cut_from_xye(self, x, y, e, x_units, selected_workspace, intensity_range=None, plot_over=False,
                          cut_ws_name=None, legend=None):
        legend = selected_workspace if legend is None else legend
        plt.errorbar(x, y, yerr=e, label=legend, hold=plot_over, marker='o', picker=picker)
        plt.ylim(*intensity_range) if intensity_range is not None else plt.autoscale()
        leg = plt.legend(fontsize='medium')
        leg.draggable()
        plt.xlabel(self._getDisplayName(x_units, self._cut_algorithm.getComment(selected_workspace)), picker=picker)
        plt.ylabel(INTENSITY_LABEL, picker=picker)
        if not plot_over:
            plt.gcf().canvas.manager.update_grid()
        if self.background is None:
            self._create_cut(cut_ws_name if cut_ws_name is not None else selected_workspace)
        plt.gcf().canvas.restore_region(self.background)
        try:
            plt.gca().draw_artist(plt.gcf().canvas.figure.get_children()[1])
            plt.gcf().canvas.blit(plt.gcf().canvas.figure.gca().clipbox)
        except AttributeError:
            plt.gcf().canvas.draw()

    def _create_cut(self, workspace):
        # don't include axis ticks in the saved background
        plt.gcf().canvas.figure.gca().xaxis.set_visible(False)
        plt.gcf().canvas.figure.gca().yaxis.set_visible(False)
        plt.gcf().canvas.draw()
        self.background = plt.gcf().canvas.copy_from_bbox(plt.gcf().canvas.figure.bbox)

        plt.gcf().canvas.figure.gca().xaxis.set_visible(True)
        plt.gcf().canvas.figure.gca().yaxis.set_visible(True)
        plt.gcf().canvas.manager.add_cut_plot(self, workspace)
        plt.gcf().canvas.draw()

    def set_icut(self, icut):
        if icut is not None:
            if hasattr(icut, 'plot_cut'):
                plt.gcf().canvas.manager.is_icut(True)
                self.icut = icut
            else:
                plt.gcf().canvas.manager.is_icut(icut)
        else:
            self.icut = None

    def get_icut(self):
        return self.icut

    def save_cut(self, params):
        self._cut_algorithm.compute_cut(*params)

    def _getDisplayName(self, axisUnits, comment=None):
        if 'DeltaE' in axisUnits:
            # Matplotlib 1.3 doesn't handle LaTeX very well. Sometimes no legend appears if we use LaTeX
            if MPL_COMPAT:
                return 'Energy Transfer ' + ('(cm-1)' if (comment and 'wavenumber' in comment) else '(meV)')
            else:
                return 'Energy Transfer ' + ('(cm$^{-1}$)' if (comment and 'wavenumber' in comment) else '(meV)')
        elif 'MomentumTransfer' in axisUnits or '|Q|' in axisUnits:
            return '|Q| (recip. Ang.)' if MPL_COMPAT else '$|Q|$ ($\mathrm{\AA}^{-1}$)'
        elif 'Degrees' in axisUnits:
            return 'Scattering Angle (degrees)' if MPL_COMPAT else r'Scattering Angle 2$\theta$ ($^{\circ}$)'
        else:
            return axisUnits

    def _generate_legend(self, workspace_name, integrated_dim, integration_start, integration_end):
        if MPL_COMPAT:
            mappings = {'DeltaE':'E', 'MomentumTransfer':'|Q|', 'Degrees':r'2Theta'}
        else:
            mappings = {'DeltaE':'E', 'MomentumTransfer':'|Q|', 'Degrees':r'2$\theta$'}
        integrated_dim = mappings[integrated_dim] if integrated_dim in mappings else integrated_dim
        return workspace_name + " " + "%.2f" % integration_start + "<" + integrated_dim + "<" + \
            "%.2f" % integration_end

    def set_workspace_provider(self, workspace_provider):
        self.workspace_provider = workspace_provider
        self._cut_algorithm.set_workspace_provider(workspace_provider)
