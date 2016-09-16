from matplotlib.colors import Normalize
from slice_plotter import SlicePlotter
import plotting.pyplot as plt


class MatplotlibSlicePlotter(SlicePlotter):
    def __init__(self, slice_algorithm):
        self._slice_algorithm = slice_algorithm
        self._colormaps = ['viridis', 'jet', 'summer', 'winter', 'coolwarm']

    def plot_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                   colourmap):
        plot_data, boundaries = self._slice_algorithm.compute_slice(selected_workspace, x_axis, y_axis, smoothing,
                                                                    norm_to_one)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        plt.imshow(plot_data, extent=boundaries, cmap=colourmap, aspect='auto', norm=norm,
                   interpolation='none', hold=False)
        plt.xlabel(x_axis.units)
        plt.ylabel(y_axis.units)
        plt.draw_all()

    def get_available_colormaps(self):
        return self._colormaps

    def get_available_axis(self, selected_workspace):
        return self._slice_algorithm.get_available_axis(selected_workspace)

    def get_axis_range(self, workspace, dimension_name):
        return self._slice_algorithm.get_axis_range(workspace, dimension_name)
