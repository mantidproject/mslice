from plotting import pyplot as plt
from matplotlib.colors import Normalize, NoNorm
from slice_plotter import SlicePlotter


class MatplotlibSlicePlotter(SlicePlotter):
    def __init__(self, slice_algorithm):
        self._slice_algorithm = slice_algorithm
        self._colormaps = ['viridis', 'jet', 'summer', 'winter']

    def display_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                      colourmap):
            plot_data = self._slice_algorithm.compute_slice(selected_workspace, x_axis, y_axis, smoothing)
            boundaries = [x_axis.start, x_axis.end, y_axis.start, y_axis.end]
            if norm_to_one:
                plot_data = self._slice_algorithm._norm_to_one(plot_data)
            norm = Normalize(vmin=intensity_start, vmax=intensity_end)
            plt.imshow(plot_data, extent=boundaries, interpolation='none', aspect='auto', cmap=colourmap, norm=norm)
            plt.xlabel(x_axis.units)
            plt.ylabel(y_axis.units)

    def get_available_colormaps(self):
        return self._colormaps

    def get_available_axis(self, selected_workspace):
        return self._slice_algorithm.get_available_axis(selected_workspace)