from cut_plotter import CutPlotter


import plotting.pyplot as plt


class MatplotlibCutPlotter(CutPlotter):
    def __init__(self, cut_algorithm):
        self._cut_algorithm = cut_algorithm

    def plot_cut(self, selected_workspace, cut_axis, integration_start, integration_end,
                 intensity_start, intensity_end, norm_to_one, plot_over):
        x,y = self._cut_algorithm.compute_cut(selected_workspace, cut_axis, integration_start, integration_end)
        plt.plot(x, y, hold=plot_over)

