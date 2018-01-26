from __future__ import (absolute_import, division, print_function)
import mslice.plotting.pyplot as plt
from mslice.app import MPL_COMPAT
from .cut_plotter import CutPlotter

INTENSITY_LABEL = 'Signal/#Events'
picker=3


class MatplotlibCutPlotter(CutPlotter):
    def __init__(self, cut_algorithm):
        self._cut_algorithm = cut_algorithm

    def plot_cut(self, selected_workspace, cut_axis, integration_start, integration_end, norm_to_one, intensity_start,
                 intensity_end, plot_over):
        x, y, e = self._cut_algorithm.compute_cut_xye(selected_workspace, cut_axis, integration_start, integration_end,
                                                      norm_to_one)
        integrated_dim = self._cut_algorithm.get_other_axis(selected_workspace, cut_axis)
        self.background = plt.gcf().canvas.copy_from_bbox(plt.gca().bbox)
        self.legend = self._generate_legend(selected_workspace, integrated_dim, integration_start, integration_end)
        plt.xlabel(self._getDisplayName(cut_axis.units, self._cut_algorithm.getComment(selected_workspace)), picker=picker)
        plt.ylabel(INTENSITY_LABEL, picker=picker)
        plt.autoscale()
        plt.ylim(intensity_start, intensity_end)
        self.line = plt.errorbar(x, y, yerr=e, label=self.legend, hold=plot_over, marker='o', picker=picker)
        leg = plt.legend(fontsize='medium')
        leg.draggable()
        # plt.gca().get_xaxis().set_animated(True)
        plt.gcf().canvas.manager.add_cut_plot(self)
        if not plot_over:
            plt.gcf().canvas.manager.update_grid()
        plt.draw_all()
        self.canvas = plt.gcf().canvas

    def update_cut(self, workspace, cut_axis, integration_start, integration_end, norm_to_one, intensity_start, intensity_end, plot_over):
        x, y, e = self._cut_algorithm.compute_cut_xye(workspace, cut_axis, integration_start, integration_end,
                                                      norm_to_one)
        self.canvas.restore_region(self.background)
        plt.errorbar(x, y, yerr=e, label=self.legend, hold=plot_over, marker='o', picker=picker)
        self.canvas.figure.gca().draw_artist(self.canvas.figure.get_children()[1])
        self.canvas.blit(self.canvas.figure.gca().clipbox)
        self.canvas.draw()

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
