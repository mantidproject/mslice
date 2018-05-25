from __future__ import (absolute_import, division, print_function)
from matplotlib.colors import Normalize

from mslice.models.cmap import allowed_cmaps
from mslice.models.labels import get_display_name, recoil_labels
from mslice.models.slice.slice_plotter import SlicePlotter
from mslice.models.slice.slice_functions import (compute_slice, sample_temperature, compute_recoil_line,
                                                 compute_chi_magnetic, compute_gdos, compute_d2sigma,
                                                 compute_powder_line, compute_chi, compute_symmetrised)
from mslice.models.workspacemanager.workspace_algorithms import get_comment
import mslice.plotting.pyplot as plt

OVERPLOT_COLORS = {1: 'b', 2: 'g', 4: 'r', 'Aluminium': 'g', 'Copper': 'm', 'Niobium': 'y', 'Tantalum': 'b'}
PICKER_TOL_PTS = 5


class MatplotlibSlicePlotter(SlicePlotter):

    def __init__(self):
        self.listener = None
        self.slice_cache = {}
        self._sample_temp_fields = []
        self.overplot_lines = {}

    def plot_slice(self, selected_ws, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                   colourmap):
        sample_temp = sample_temperature(selected_ws, self._sample_temp_fields)
        plot_data, boundaries = compute_slice(selected_ws, x_axis, y_axis, norm_to_one)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        self._cache_slice(plot_data, selected_ws, boundaries, colourmap, norm, sample_temp, x_axis, y_axis)
        if selected_ws not in self.overplot_lines:
            self.overplot_lines[selected_ws] = {}
        self.show_scattering_function(selected_ws)
        fig_canvas = plt.gcf().canvas
        fig_canvas.set_window_title(selected_ws)
        fig_canvas.manager.add_slice_plot(self, selected_ws)
        fig_canvas.manager.update_grid()
        plt.draw_all()

    def _cache_slice(self, plot_data, ws, boundaries, colourmap, norm, sample_temp, x_axis, y_axis):
        self.slice_cache[ws] = {'plot_data': plot_data, 'boundaries': boundaries, 'colourmap': colourmap,
                                'norm': norm, 'sample_temp': sample_temp}
        if x_axis.units == 'MomentumTransfer' or x_axis.units == 'Degrees' or x_axis.units == '|Q|':
            self.slice_cache[ws]['momentum_axis'] = x_axis
            self.slice_cache[ws]['energy_axis'] = y_axis
            self.slice_cache[ws]['rotated'] = False
        else:
            self.slice_cache[ws]['momentum_axis'] = y_axis
            self.slice_cache[ws]['energy_axis'] = x_axis
            self.slice_cache[ws]['rotated'] = True

    def _show_plot(self, workspace_name, plot_data, extent, colourmap, norm, momentum_axis, energy_axis):
        plt.clf()
        image = plt.imshow(plot_data, extent=extent, cmap=colourmap, aspect='auto', norm=norm,
                           interpolation='none')
        plot_axes.set_title(workspace_name, picker=PICKER_TOL_PTS)
        plot_axes = plt.gca()
        if self.slice_cache[workspace_name]['rotated']:
            x_axis = energy_axis
            y_axis = momentum_axis
        else:
            x_axis = momentum_axis
            y_axis = energy_axis
        comment = get_comment(str(workspace_name))
        # labels
        plot_axes.set_xlabel(get_display_name(x_axis.units, comment), picker=PICKER_TOL_PTS)
        plot_axes.set_ylabel(get_display_name(y_axis.units, comment), picker=PICKER_TOL_PTS)
        plot_axes.set_xlim(x_axis.start)
        plot_axes.set_ylim(y_axis.start)
        plot_axes.get_xaxis().set_units(x_axis.units)
        plot_axes.get_yaxis().set_units(y_axis.units)

        # colorbar
        cb = plt.colorbar(image, ax=plot_axes)
        cb.set_label('Intensity (arb. units)', labelpad=20, rotation=270, picker=PICKER_TOL_PTS)
        plt.gcf().canvas.draw_idle()
        plt.show()

    def show_scattering_function(self, workspace):
        cached_slice = self.slice_cache[workspace]
        self._show_plot(workspace, cached_slice['plot_data'][0], cached_slice['boundaries'], cached_slice['colourmap'],
                        cached_slice['norm'], cached_slice['momentum_axis'], cached_slice['energy_axis'])

    def show_dynamical_susceptibility(self, workspace):
        cached_slice = self.slice_cache[workspace]
        if cached_slice['plot_data'][1] is None:
            self.compute_chi(workspace)
        self._show_plot(workspace, cached_slice['plot_data'][1], cached_slice['boundaries'], cached_slice['colourmap'],
                        cached_slice['norm'], cached_slice['momentum_axis'], cached_slice['energy_axis'])

    def show_dynamical_susceptibility_magnetic(self, workspace):
        cached_slice = self.slice_cache[workspace]
        if cached_slice['plot_data'][2] is None:
            self.compute_chi_magnetic(workspace)
        self._show_plot(workspace, cached_slice['plot_data'][2], cached_slice['boundaries'], cached_slice['colourmap'],
                        cached_slice['norm'], cached_slice['momentum_axis'], cached_slice['energy_axis'])
        plt.gcf().get_axes()[1].set_ylabel('chi\'\'(Q,E) |F(Q)|$^2$ ($mu_B$ $meV^{-1} sr^{-1} f.u.^{-1}$)',
                                           rotation=270, labelpad=20)

    def show_d2sigma(self, workspace):
        cached_slice = self.slice_cache[workspace]
        if cached_slice['plot_data'][3] is None:
            self.compute_d2sigma(workspace)
        self._show_plot(workspace, cached_slice['plot_data'][3], cached_slice['boundaries'], cached_slice['colourmap'],
                        cached_slice['norm'], cached_slice['momentum_axis'], cached_slice['energy_axis'])

    def show_symmetrised(self, workspace):
        cached_slice = self.slice_cache[workspace]
        if cached_slice['plot_data'][4] is None:
            self.compute_symmetrised(workspace)
        self._show_plot(workspace, cached_slice['plot_data'][4], cached_slice['boundaries'], cached_slice['colourmap'],
                        cached_slice['norm'], cached_slice['momentum_axis'], cached_slice['energy_axis'])

    def show_gdos(self, workspace):
        cached_slice = self.slice_cache[workspace]
        if cached_slice['plot_data'][5] is None:
            self.compute_gdos(workspace)
        self._show_plot(workspace, cached_slice['plot_data'][5], cached_slice['boundaries'], cached_slice['colourmap'],
                        cached_slice['norm'], cached_slice['momentum_axis'], cached_slice['energy_axis'])

    def add_overplot_line(self, workspace, key, recoil, extra_info):
        if recoil:  # key is relative mass
            label = recoil_labels[key] if key in recoil_labels else \
                'Relative mass ' + str(key)
        else:  # key is element name
            label = key
        if key in self.overplot_lines[workspace]:
            line = self.overplot_lines[workspace][key]
            line.set_linestyle('-')  # make visible
            line.set_label(label)  # add to legend
            line.set_markersize(6)  # show markers - 6.0 is default size
            if line not in plt.gca().get_children():
                plt.gca().add_artist(line)
        else:
            momentum_axis = self.slice_cache[workspace]['momentum_axis']
            if recoil:
                x, y = compute_recoil_line(workspace, momentum_axis, key)
            else:
                x, y = compute_powder_line(workspace, momentum_axis, key, cif_file=extra_info)
            color = overplot_colors[key] if key in overplot_colors else 'c'
            if self.slice_cache[workspace]['rotated']:
                self.overplot_lines[workspace][key] = plt.gca().plot(y, x, color=color, label=label,
                                                                     alpha=.7, picker=picker)[0]
            else:
                self.overplot_lines[workspace][key] = plt.gca().plot(x, y, color=color, label=label,
                                                                     alpha=.7, picker=picker)[0]

    def hide_overplot_line(self, workspace, key):
        if key in self.overplot_lines[workspace]:
            line = self.overplot_lines[workspace][key]
            line.set_linestyle('')
            line.set_label('')
            line.set_markersize(0)

    def add_sample_temperature_field(self, field_name):
        self._sample_temp_fields.append(field_name)

    def update_sample_temperature(self, workspace):
        temp = sample_temperature(workspace, self._sample_temp_fields)
        self.slice_cache[workspace]['sample_temp'] = temp

    def set_sample_temperature(self, workspace, temp):
        self.slice_cache[workspace]['sample_temp'] = temp

    def sample_temperature(self, workspace):
        cached_slice = self.slice_cache[workspace]
        sample_temp = cached_slice['sample_temp']
        if sample_temp is not None:
            return sample_temp
        else:
            raise ValueError('sample temperature not found')

    def compute_chi(self, workspace):
        cached_slice = self.slice_cache[workspace]
        cached_slice['plot_data'][1] = compute_chi(cached_slice['plot_data'][0], self.sample_temperature(workspace),
                                                   cached_slice['energy_axis'], cached_slice['rotated'])

    def compute_chi_magnetic(self, workspace):
        cached_slice = self.slice_cache[workspace]
        if cached_slice['plot_data'][1] is None:
            self.compute_chi(workspace)
        cached_slice['plot_data'][2] = compute_chi_magnetic(cached_slice['plot_data'][1])

    def compute_d2sigma(self, workspace):
        cached_slice = self.slice_cache[workspace]
        cached_slice['plot_data'][3] = compute_d2sigma(cached_slice['plot_data'][0],
                                                       workspace, cached_slice['energy_axis'], cached_slice['rotated'])

    def compute_symmetrised(self, workspace):
        cached_slice = self.slice_cache[workspace]
        cached_slice['plot_data'][4] = compute_symmetrised(cached_slice['plot_data'][0],
                                                           self.sample_temperature(workspace),
                                                           cached_slice['energy_axis'], cached_slice['rotated'])

    def compute_gdos(self, workspace):
        cached_slice = self.slice_cache[workspace]
        cached_slice['plot_data'][5] = compute_gdos(cached_slice['plot_data'][0], self.sample_temperature(workspace),
                                                    cached_slice['momentum_axis'], cached_slice['energy_axis'],
                                                    cached_slice['rotated'])

    def get_available_colormaps(self):
        return allowed_cmaps()

    def get_recoil_label(self, key):
        return recoil_labels[key]

    def update_displayed_workspaces(self):
        self.listener.update_workspaces()
