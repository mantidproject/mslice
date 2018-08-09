from __future__ import (absolute_import, division, print_function)
from matplotlib.colors import Normalize

from mslice.models.cmap import allowed_cmaps
from mslice.models.labels import get_display_name, recoil_labels
from mslice.models.slice.slice_plotter import SlicePlotter
from mslice.models.slice.slice_cache import SliceCache
from mslice.models.slice.slice_functions import (compute_slice, sample_temperature, compute_recoil_line,
                                                 compute_powder_line)
from mslice.models.workspacemanager.workspace_algorithms import get_comment, get_workspace_handle
import mslice.plotting.pyplot as plt
from mslice.util.mantid.mantid_algorithms import Transpose

OVERPLOT_COLORS = {1: 'b', 2: 'g', 4: 'r', 'Aluminium': 'g', 'Copper': 'm', 'Niobium': 'y', 'Tantalum': 'b'}
PICKER_TOL_PTS = 5


class MatplotlibSlicePlotter(SlicePlotter):

    def __init__(self):
        self.listener = None
        self.slice_cache = {}
        self._sample_temp_fields = []

    def plot_slice(self, selected_ws, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                   colourmap):
        selected_ws = get_workspace_handle(selected_ws)
        sample_temp = sample_temperature(selected_ws, self._sample_temp_fields)
        slice = compute_slice(selected_ws, x_axis, y_axis, norm_to_one)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        self._cache_slice(slice, colourmap, norm, sample_temp, x_axis, y_axis)
        self.show_scattering_function(selected_ws.name)
        fig_canvas = plt.gcf().canvas
        fig_canvas.set_window_title(selected_ws.name)
        fig_canvas.manager.add_slice_plot(self, selected_ws.name)
        fig_canvas.manager.update_grid()
        plt.draw_all()

    def _cache_slice(self, slice, colourmap, norm, sample_temp, x_axis, y_axis):
        rotated = x_axis.units not in ['MomentumTransfer', 'Degrees', '|Q|']
        q_axis = y_axis if rotated else x_axis
        e_axis = x_axis if rotated else y_axis
        self.slice_cache[slice.name[2:]] = SliceCache(slice, colourmap, norm, sample_temp, q_axis, e_axis, rotated)

    @plt.set_category(plt.CATEGORY_SLICE)
    def _show_plot(self, slice_cache, workspace):
        # Do not call plt.gcf() here as the overplot Line1D objects have been cached and they
        # must be redrawn on the same Axes instance
        cur_fig = plt.gcf()
        cur_fig.clf()
        ax = cur_fig.add_subplot(111, projection='mantid')
        if not workspace.is_PSD and not slice_cache.rotated:
            workspace = Transpose(OutputWorkspace=workspace.name, InputWorkspace=workspace, store=False)
        image = ax.pcolormesh(workspace.raw_ws, cmap=slice_cache.colourmap, norm=slice_cache.norm)
        ax.set_title(workspace.name[2:], picker=PICKER_TOL_PTS)
        x_axis = slice_cache.energy_axis if slice_cache.rotated else slice_cache.momentum_axis
        y_axis = slice_cache.momentum_axis if slice_cache.rotated else slice_cache.energy_axis
        comment = get_comment(str(workspace.name))
        ax.get_xaxis().set_units(x_axis.units)
        ax.get_yaxis().set_units(y_axis.units)
        # labels
        ax.set_xlabel(get_display_name(x_axis.units, comment), picker=PICKER_TOL_PTS)
        ax.set_ylabel(get_display_name(y_axis.units, comment), picker=PICKER_TOL_PTS)
        ax.set_xlim(x_axis.start)
        ax.set_ylim(y_axis.start)
        cb = plt.colorbar(image, ax=ax)
        cb.set_label('Intensity (arb. units)', labelpad=20, rotation=270, picker=PICKER_TOL_PTS)
        cur_fig.canvas.draw_idle()
        cur_fig.show()

    def show_scattering_function(self, workspace_name):
        slice_cache = self.slice_cache[workspace_name]
        self._show_plot(slice_cache, slice_cache.scattering_function)

    def show_dynamical_susceptibility(self, workspace_name):
        slice_cache = self.slice_cache[workspace_name]
        self._show_plot(slice_cache, slice_cache.chi)

    def show_dynamical_susceptibility_magnetic(self, workspace_name):
        slice_cache = self.slice_cache[workspace_name]
        self._show_plot(slice_cache, slice_cache.chi_magnetic)
        plt.gcf().get_axes()[1].set_ylabel('chi\'\'(Q,E) |F(Q)|$^2$ ($mu_B$ $meV^{-1} sr^{-1} f.u.^{-1}$)',
                                           rotation=270, labelpad=20)

    def show_d2sigma(self, workspace_name):
        slice_cache = self.slice_cache[workspace_name]
        self._show_plot(slice_cache, slice_cache.d2sigma)

    def show_symmetrised(self, workspace_name):
        slice_cache = self.slice_cache[workspace_name]
        self._show_plot(slice_cache, slice_cache.symmetrised)

    def show_gdos(self, workspace_name):
        slice_cache = self.slice_cache[workspace_name]
        self._show_plot(slice_cache, slice_cache.gdos)

    def add_overplot_line(self, workspace_name, key, recoil, cif=None):
        cache = self.slice_cache[workspace_name]
        if recoil:  # key is relative mass
            label = recoil_labels[key] if key in recoil_labels else 'Relative mass ' + str(key)
        else:  # key is element name
            label = key
        if recoil:
            x, y = compute_recoil_line(workspace_name, cache.momentum_axis, key)
        else:
            x, y = compute_powder_line(workspace_name, cache.momentum_axis, key, cif_file=cif)
        self.plot_overplot_line(x, y, key, label, cache)

    def hide_overplot_line(self, workspace, key):
        cache = self.slice_cache[workspace]
        if key in cache.overplot_lines:
            line = cache.overplot_lines.pop(key)
            plt.gca().lines.remove(line)

    def plot_overplot_line(self, x, y, key, label, cache):
        color = OVERPLOT_COLORS[key] if key in OVERPLOT_COLORS else 'c'
        if cache.rotated:
            cache.overplot_lines[key] = plt.gca().plot(y, x, color=color, label=label, alpha=.7,
                                                       picker=PICKER_TOL_PTS)[0]
        else:
            cache.overplot_lines[key] = plt.gca().plot(x, y, color=color, label=label, alpha=.7,
                                                       picker=PICKER_TOL_PTS)[0]

    def add_sample_temperature_field(self, field_name):
        self._sample_temp_fields.append(field_name)

    def update_sample_temperature(self, workspace_name):
        temp = sample_temperature(workspace_name, self._sample_temp_fields)
        self.set_sample_temperature(workspace_name, temp)

    def set_sample_temperature(self, workspace_name, temp):
        self.slice_cache[workspace_name].sample_temp = temp

    def get_available_colormaps(self):
        return allowed_cmaps()

    def update_displayed_workspaces(self):
        self.listener.update_workspaces()
