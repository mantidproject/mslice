import mslice.plotting.pyplot as plt
from .cut_plotter import CutPlotter

INTENSITY_LABEL = 'Signal/#Events'


class MatplotlibCutPlotter(CutPlotter):
    def __init__(self, cut_algorithm):
        self._cut_algorithm = cut_algorithm

    def plot_cut(self, selected_workspace, cut_axis, integration_start, integration_end, norm_to_one, intensity_start,
                 intensity_end, plot_over):
        x, y, e = self._cut_algorithm.compute_cut_xye(selected_workspace, cut_axis, integration_start, integration_end,
                                                      norm_to_one)
        integrated_dim = self._cut_algorithm.get_other_axis(selected_workspace, cut_axis)
        legend = self._generate_legend(selected_workspace, integrated_dim, integration_start, integration_end)
        plt.errorbar(x, y, yerr=e, label=legend, hold=plot_over)
        plt.legend()
        plt.xlabel(self._getDisplayName(cut_axis.units, self._cut_algorithm.getComment(selected_workspace)))
        plt.ylabel(INTENSITY_LABEL)
        plt.autoscale()
        plt.ylim(intensity_start, intensity_end)
        plt.draw_all()

    def _getDisplayName(self, axisUnits, comment=None):
        if 'DeltaE' in axisUnits:
            return 'Energy Transfer ' + ('(cm$^{-1}$)' if (comment and 'wavenumber' in comment) else '(meV)')
        elif 'MomentumTransfer' in axisUnits or '|Q|' in axisUnits:
            return '$|Q|$ ($\mathrm{\AA}^{-1}$)'
        elif 'Degrees' in axisUnits:
            return r'Scattering Angle 2$\theta$ ($^{\circ}$)'
        else:
            return axisUnits

    def _generate_legend(self, workspace_name, integrated_dim, integration_start, integration_end):
        mappings = {'DeltaE':'E', 'MomentumTransfer':'|Q|', 'Degrees':r'2$\theta$'}
        integrated_dim = mappings[integrated_dim] if integrated_dim in mappings.keys() else integrated_dim
        return workspace_name + " " + "%.2f" % integration_start + "<" + integrated_dim + "<" + \
            "%.2f" % integration_end
