from __future__ import (absolute_import, division, print_function)
import numpy as np
from matplotlib.colors import Normalize
from mantid.simpleapi import AnalysisDataService
from .slice_plotter import SlicePlotter
import mslice.plotting.pyplot as plt
from mslice.app import MPL_COMPAT


class MatplotlibSlicePlotter(SlicePlotter):
    def __init__(self, slice_algorithm):
        self._slice_algorithm = slice_algorithm
        self._colormaps = ['jet', 'summer', 'winter', 'coolwarm']
        if not MPL_COMPAT:
            self._colormaps.insert(0, 'viridis')
        self.slice_cache = {}
        self._sample_temp_fields = []

    def plot_slice(self, selected_ws, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                   colourmap):
        sample_temp = self.sample_temperature(selected_ws)
        plot_data, boundaries = self._slice_algorithm.compute_slice(selected_ws, x_axis, y_axis,
                                                                    smoothing, norm_to_one, sample_temp)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        self.slice_cache[selected_ws] = {'plot_data': plot_data, 'boundaries': boundaries, 'x_axis': x_axis,
                                         'y_axis': y_axis, 'colourmap': colourmap, 'norm': norm,
                                         'sample_temp': sample_temp}
        self.show_scattering_function(selected_ws)
        plt.gcf().canvas.set_window_title(selected_ws)
        plt.gcf().canvas.manager.add_slice_plotter(self)
        plt.draw_all()

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

    def _show_plot(self, workspace_name, plot_data, extent, colourmap, norm, x_axis, y_axis):
        plt.imshow(plot_data, extent=extent, cmap=colourmap, aspect='auto', norm=norm,
                   interpolation='none', hold=False)
        plt.title(workspace_name)
        comment = self._slice_algorithm.getComment(str(workspace_name))
        plt.xlabel(self._getDisplayName(x_axis.units, comment))
        plt.ylabel(self._getDisplayName(y_axis.units, comment))
        plt.gcf().get_axes()[1].set_ylabel('Intensity (arb. units)', labelpad=20, rotation=270)

    def show_scattering_function(self, workspace):
        slice_cache = self.slice_cache[workspace]
        self._show_plot(workspace, slice_cache['plot_data'][0], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])

    def show_dynamical_susceptibility(self, workspace):
        slice_cache = self.slice_cache[workspace]
        if slice_cache['plot_data'][1] is None:
            raise ValueError('plot_data not calculated')
        self._show_plot(workspace, slice_cache['plot_data'][1], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])

    def show_dynamical_susceptibility_magnetic(self, workspace):
        slice_cache = self.slice_cache[workspace]
        if slice_cache['plot_data'][1] is None:
            raise ValueError('plot_data not calculated')
        self._show_plot(workspace, slice_cache['plot_data'][2], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])
        plt.gcf().get_axes()[1].set_ylabel('chi\'\'(Q,E) |F(Q)|$^2$ ($mu_B$ $meV^{-1} sr^{-1} f.u.^{-1}$)',
                                           rotation=270, labelpad=20)

    def add_sample_temperature_field(self, field_name):
        self._sample_temp_fields.append(field_name)

    def update_sample_temperature(self, workspace):
        temp = self.sample_temperature(str(workspace))
        self.slice_cache[workspace]['sample_temp'] = temp
        self.slice_cache[workspace]['plot_data'][1] = self._slice_algorithm.compute_chi(
            self.slice_cache[workspace]['plot_data'][0], temp, self.slice_cache[workspace]['y_axis'])
        self.slice_cache[workspace]['plot_data'][2] = self._slice_algorithm.compute_chi_magnetic(
            self.slice_cache[workspace]['plot_data'][1])

    def sample_temperature(self, ws_name):
        if ws_name[-3:] == '_QE':
            ws_name = ws_name[:-3]
        ws = AnalysisDataService[ws_name]
        sample_temp = None
        for field_name in self._sample_temp_fields:
            try:
                sample_temp = ws.run().getLogData(field_name).value
            except RuntimeError:
                pass
        try:
            float(sample_temp)
        except (ValueError, TypeError):
            pass
        else:
            return sample_temp
        if isinstance(sample_temp, str):
            sample_temp = self.get_sample_temperature_from_string(sample_temp)
        if isinstance(sample_temp, np.ndarray) or isinstance(sample_temp, list):
            sample_temp = np.mean(sample_temp)
        return sample_temp

    def get_sample_temperature_from_string(self, string):
        pos_k = string.find('K')
        if pos_k == -1:
            return None
        k_string = string[pos_k - 3:pos_k]
        sample_temp = float(''.join(c for c in k_string if c.isdigit()))
        return sample_temp

    def get_available_colormaps(self):
        return self._colormaps

    def get_available_axis(self, selected_workspace):
        return self._slice_algorithm.get_available_axis(selected_workspace)

    def get_axis_range(self, workspace, dimension_name):
        return self._slice_algorithm.get_axis_range(workspace, dimension_name)

    def set_workspace_provider(self, workspace_provider):
        self._slice_algorithm.set_workspace_provider(workspace_provider)
