from matplotlib.colors import Normalize

from .slice_plotter import SlicePlotter
import mslice.plotting.pyplot as plt

class MatplotlibSlicePlotter(SlicePlotter):
    def __init__(self, slice_algorithm):
        self._slice_algorithm = slice_algorithm
        import matplotlib
        ver = float('.'.join(matplotlib.__version__.split('.')[:2]))
        self._colormaps = ['jet', 'summer', 'winter', 'coolwarm']
        if ver >= 1.5:
            self._colormaps.insert(0, 'viridis')

    def plot_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                   colourmap):
        plot_data, boundaries = self._slice_algorithm.compute_slice(selected_workspace, x_axis, y_axis, smoothing,
                                                                    norm_to_one)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        plt.imshow(plot_data, extent=boundaries, cmap=colourmap, aspect='auto', norm=norm,
                   interpolation='none', hold=False)
        comment = self._slice_algorithm.getComment(selected_workspace)
        plt.xlabel(self._getDisplayName(x_axis.units, comment))
        plt.ylabel(self._getDisplayName(y_axis.units, comment))
        plt.title(selected_workspace)
        plt.draw_all()

    def _getDisplayName(self, axisUnits, comment=None):
        if 'DeltaE' in axisUnits:
            return 'Energy Transfer ' + ('(cm$^{-1}$)' if (comment and 'wavenumber' in comment) else '(meV)')
        elif 'MomentumTransfer' in axisUnits or '|Q|' in axisUnits:
            return '$|Q|$ ($\mathrm{\AA}^{-1}$)'
        elif 'Degrees' in axisUnits:
            return 'Scattering Angle 2$\theta$ ($^{\circ}$)'
        else:
            return axisUnits

    def get_available_colormaps(self):
        return self._colormaps

    def get_available_axis(self, selected_workspace):
        return self._slice_algorithm.get_available_axis(selected_workspace)

    def get_axis_range(self, workspace, dimension_name):
        return self._slice_algorithm.get_axis_range(workspace, dimension_name)

    def set_workspace_provider(self, workspace_provider):
        self._slice_algorithm.set_workspace_provider(workspace_provider)
