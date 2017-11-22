from __future__ import (absolute_import, division, print_function)
from matplotlib.colors import Normalize
from .slice_plotter import SlicePlotter
import mslice.plotting.pyplot as plt
from mslice.app import MPL_COMPAT

recoil_colors={1:'b', 2:'g', 4:'r'}
recoil_labels={1:'Hydrogen', 2:'Deuterium', 4:'Helium'}
powder_colors={'aluminium': 'g', 'copper':'m', 'niobium':'y', 'tantalum':'b'}


class MatplotlibSlicePlotter(SlicePlotter):
    def __init__(self, slice_algorithm):
        self._slice_algorithm = slice_algorithm
        self._colormaps = ['jet', 'summer', 'winter', 'coolwarm']
        if not MPL_COMPAT:
            self._colormaps.insert(0, 'viridis')
        self.slice_cache = {}
        self._sample_temp_fields = []
        self.recoil_lines = {}
        self.powder_lines = {}

    def plot_slice(self, selected_ws, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                   colourmap):
        sample_temp = self._slice_algorithm.sample_temperature(selected_ws, self._sample_temp_fields)
        plot_data, boundaries = self._slice_algorithm.compute_slice(selected_ws, x_axis, y_axis,
                                                                    smoothing, norm_to_one)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        self.slice_cache[selected_ws] = {'plot_data': plot_data, 'boundaries': boundaries, 'x_axis': x_axis,
                                         'y_axis': y_axis, 'colourmap': colourmap, 'norm': norm,
                                         'sample_temp': sample_temp, 'boltzmann_dist': None}
        self.recoil_lines[selected_ws] = {}
        self.powder_lines[selected_ws] = {}
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
        plt.xlim(x_axis.start)
        plt.ylim(y_axis.start)
        plt.gcf().get_axes()[1].set_ylabel('Intensity (arb. units)', labelpad=20, rotation=270)

    def show_scattering_function(self, workspace):
        slice_cache = self.slice_cache[workspace]
        self._show_plot(workspace, slice_cache['plot_data'][0], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])

    def show_dynamical_susceptibility(self, workspace):
        slice_cache = self.slice_cache[workspace]
        if slice_cache['plot_data'][1] is None:
            self.compute_chi(workspace)
        self._show_plot(workspace, slice_cache['plot_data'][1], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])

    def show_dynamical_susceptibility_magnetic(self, workspace):
        slice_cache = self.slice_cache[workspace]
        if slice_cache['plot_data'][2] is None:
            self.compute_chi_magnetic(workspace)
        self._show_plot(workspace, slice_cache['plot_data'][2], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])
        plt.gcf().get_axes()[1].set_ylabel('chi\'\'(Q,E) |F(Q)|$^2$ ($mu_B$ $meV^{-1} sr^{-1} f.u.^{-1}$)',
                                           rotation=270, labelpad=20)

    def show_d2sigma(self, workspace):
        slice_cache = self.slice_cache[workspace]
        if slice_cache['plot_data'][2] is None:
            self.compute_d2sigma(workspace)
        self._show_plot(workspace, slice_cache['plot_data'][3], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])

    def show_symmetrised(self, workspace):
        slice_cache = self.slice_cache[workspace]
        if slice_cache['plot_data'][4] is None:
            self.compute_symmetrised(workspace)
        self._show_plot(workspace, slice_cache['plot_data'][4], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])

    def show_gdos(self, workspace):
        slice_cache = self.slice_cache[workspace]
        if slice_cache['plot_data'][5] is None:
            self.compute_gdos(workspace)
        self._show_plot(workspace, slice_cache['plot_data'][5], slice_cache['boundaries'], slice_cache['colourmap'],
                        slice_cache['norm'], slice_cache['x_axis'], slice_cache['y_axis'])

    def add_recoil_line(self, workspace, relative_mass):
        label = recoil_labels[relative_mass] if relative_mass in recoil_labels else 'Relative mass '+ str(relative_mass)
        if relative_mass in self.recoil_lines[workspace]:
            line = self.recoil_lines[workspace][relative_mass]
            line.set_linestyle('-')  # make visible
            line.set_label(label)  # add to legend
        else:
            x_axis = self.slice_cache[workspace]['x_axis']
            x, y = self._slice_algorithm.compute_recoil_line(x_axis, relative_mass)
            color = recoil_colors[relative_mass] if relative_mass in recoil_colors else 'c'
            self.recoil_lines[workspace][relative_mass] = plt.gca().plot(x, y, color, label=label, alpha=.7)[0]

    def hide_recoil_line(self, workspace, relative_mass):
        if relative_mass in self.recoil_lines[workspace]:
            line = self.recoil_lines[workspace][relative_mass]
            line.set_linestyle('')
            line.set_label('')

    def add_powder_line(self, workspace, element):
        if element in self.powder_lines[workspace]:
            line = self.powder_lines[workspace][element]
            line.set_linestyle('-')  # make visible
            line.set_label(element)  # add to legend
        else:
            x_axis = self.slice_cache[workspace]['x_axis']
            y_axis = self.slice_cache[workspace]['x_axis']
            x, y = self._slice_algorithm.compute_powder_line(workspace, x_axis, element)
            color = powder_colors[element]
            self.recoil_lines[workspace][element] = plt.gca().plot(x, y, color=color, label=element, alpha=.7)

    def hide_powder_line(self, workspace, element):
        if element in self.recoil_lines[workspace]:
            lines = self.recoil_lines[workspace][element]
            for line in lines:
                line.set_linestyle('')
                line.set_label('')

    def add_sample_temperature_field(self, field_name):
        self._sample_temp_fields.append(field_name)

    def update_sample_temperature(self, workspace):
        temp = self._slice_algorithm.sample_temperature(workspace, self._sample_temp_fields)
        self.slice_cache[workspace]['sample_temp'] = temp

    def compute_boltzmann_dist(self, workspace):
        if self.slice_cache[workspace]['sample_temp'] is None:
            raise ValueError('sample temperature not found')
        self.slice_cache[workspace]['boltzmann_dist'] = self._slice_algorithm.compute_boltzmann_dist(
            self.slice_cache[workspace]['sample_temp'], self.slice_cache[workspace]['y_axis'])

    def compute_chi(self, workspace):
        if self.slice_cache[workspace]['boltzmann_dist'] is None:
            self.compute_boltzmann_dist(workspace)
        self.slice_cache[workspace]['plot_data'][1] = self._slice_algorithm.compute_chi(
            self.slice_cache[workspace]['plot_data'][0], self.slice_cache[workspace]['boltzmann_dist'],
            self.slice_cache[workspace]['y_axis'])

    def compute_chi_magnetic(self, workspace):
        if self.slice_cache[workspace]['plot_data'][1] is None:
            self.compute_chi(workspace)
        self.slice_cache[workspace]['plot_data'][2] = self._slice_algorithm.compute_chi_magnetic(
            self.slice_cache[workspace]['plot_data'][1])

    def compute_d2sigma(self, workspace):
        self.slice_cache[workspace]['plot_data'][3] = self._slice_algorithm.compute_d2sigma(
            self.slice_cache[workspace]['plot_data'][0], workspace, self.slice_cache[workspace]['y_axis'])

    def compute_symmetrised(self, workspace):
        if self.slice_cache[workspace]['boltzmann_dist'] is None:
            self.compute_boltzmann_dist(workspace)
        self.slice_cache[workspace]['plot_data'][4] = self._slice_algorithm.compute_symmetrised(
            self.slice_cache[workspace]['plot_data'][0], self.slice_cache[workspace]['boltzmann_dist'],
            self.slice_cache[workspace]['y_axis'])

    def compute_gdos(self, workspace):
        if self.slice_cache[workspace]['boltzmann_dist'] is None:
            self.compute_boltzmann_dist(workspace)
        self.slice_cache[workspace]['plot_data'][5] = self._slice_algorithm.compute_gdos(
            self.slice_cache[workspace]['plot_data'][0], self.slice_cache[workspace]['boltzmann_dist'],
            self.slice_cache[workspace]['x_axis'], self.slice_cache[workspace]['y_axis'])

    def get_available_colormaps(self):
        return self._colormaps

    def get_available_axis(self, selected_workspace):
        return self._slice_algorithm.get_available_axis(selected_workspace)

    def get_axis_range(self, workspace, dimension_name):
        return self._slice_algorithm.get_axis_range(workspace, dimension_name)

    def set_workspace_provider(self, workspace_provider):
        self._slice_algorithm.set_workspace_provider(workspace_provider)
