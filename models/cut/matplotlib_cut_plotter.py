from cut_plotter import CutPlotter
import plotting.pyplot as plt

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
        plt.xlabel(cut_axis.units)
        plt.ylabel(INTENSITY_LABEL)
        plt.autoscale()
        plt.ylim(intensity_start, intensity_end)
        plt.draw_all()

    def _generate_legend(self, workspace_name, integrated_dim, integration_start, integration_end):
        if integrated_dim == 'DeltaE':
            integrated_dim = 'e'
        return workspace_name + "    " + "%.2f" % integration_start + "<" + integrated_dim + "<" + \
               "%.2f" % integration_end
