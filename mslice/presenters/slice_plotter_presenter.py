from matplotlib.colors import Normalize

from mslice.models.slice.slice_functions import (compute_slice, sample_temperature, compute_recoil_line,
                                                 compute_powder_line)
from mslice.models.cmap import ALLOWED_CMAPS
from mslice.models.slice.slice import Slice
from mslice.models.labels import is_momentum, is_twotheta
from mslice.views.slice_plotter import (set_colorbar_label, plot_cached_slice, remove_line,
                                        plot_overplot_line, create_slice_figure)
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.presenters.presenter_utility import PresenterUtility


class SlicePlotterPresenter(PresenterUtility):

    def __init__(self):
        self._main_presenter = None
        self._slice_cache = {}
        self._sample_temp_fields = []

    def plot_slice(self, selected_ws, x_axis, y_axis, intensity_start, intensity_end, norm_to_one, colourmap):
        workspace = get_workspace_handle(selected_ws)
        self.create_slice(workspace, x_axis, y_axis, intensity_start, intensity_end, norm_to_one, colourmap)
        self.plot_from_cache(workspace)

    def create_slice(self, selected_ws, x_axis, y_axis, intensity_start, intensity_end, norm_to_one, colourmap):
        sample_temp = sample_temperature(selected_ws, self._sample_temp_fields)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        slice = compute_slice(selected_ws, x_axis, y_axis, norm_to_one)
        self._cache_slice(slice, colourmap, norm, sample_temp, x_axis, y_axis)
        self._slice_cache[selected_ws.name].norm_to_one = norm_to_one
        return slice

    def plot_from_cache(self, workspace):
        ws_name = workspace.name.lstrip('__')
        create_slice_figure(ws_name, self)
        self.show_scattering_function(ws_name)

    def change_intensity(self, workspace_name, intensity_start, intensity_end):
        workspace_name = workspace_name.lstrip('__')
        intensity_start, intensity_end = self.validate_intensity(intensity_start, intensity_end)
        norm = Normalize(vmin=intensity_start, vmax=intensity_end)
        self._slice_cache[workspace_name].norm = norm

    def change_colourmap(self, workspace_name, colourmap):
        if colourmap in ALLOWED_CMAPS:
            workspace_name = workspace_name.lstrip('__')
            self._slice_cache[workspace_name].colourmap = colourmap
        else:
            raise ValueError('colourmap not recognised')

    def update_displayed_workspaces(self):
        self._main_presenter.update_displayed_workspaces()

    def _cache_slice(self, slice, colourmap, norm, sample_temp, x_axis, y_axis):
        rotated = not is_twotheta(x_axis.units) and not is_momentum(x_axis.units)
        (q_axis, e_axis) = (x_axis, y_axis) if not rotated else (y_axis, x_axis)
        self._slice_cache[slice.name[2:]] = Slice(slice, colourmap, norm, sample_temp, q_axis, e_axis, rotated)

    def get_slice_cache(self, workspace):
        return self._slice_cache[workspace.name[2:] if hasattr(workspace, 'name') else workspace]

    def show_scattering_function(self, workspace_name):
        slice_cache = self._slice_cache[workspace_name]
        plot_cached_slice(slice_cache, slice_cache.scattering_function)

    def show_dynamical_susceptibility(self, workspace_name):
        slice_cache = self._slice_cache[workspace_name]
        plot_cached_slice(slice_cache, slice_cache.chi)

    def show_dynamical_susceptibility_magnetic(self, workspace_name):
        slice_cache = self._slice_cache[workspace_name]
        plot_cached_slice(slice_cache, slice_cache.chi_magnetic)
        set_colorbar_label('chi\'\'(Q,E) |F(Q)|$^2$ ($mu_B$ $meV^{-1} sr^{-1} f.u.^{-1}$)')

    def show_d2sigma(self, workspace_name):
        slice_cache = self._slice_cache[workspace_name]
        plot_cached_slice(slice_cache, slice_cache.d2sigma)

    def show_symmetrised(self, workspace_name):
        slice_cache = self._slice_cache[workspace_name]
        plot_cached_slice(slice_cache, slice_cache.symmetrised)

    def show_gdos(self, workspace_name):
        slice_cache = self._slice_cache[workspace_name]
        plot_cached_slice(slice_cache, slice_cache.gdos)

    def hide_overplot_line(self, workspace, key):
        cache = self._slice_cache[workspace]
        if key in cache.overplot_lines:
            line = cache.overplot_lines.pop(key)
            remove_line(line)

    def add_overplot_line(self, workspace_name, key, recoil, cif=None):
        cache = self._slice_cache[workspace_name]
        if recoil:
            x, y = compute_recoil_line(workspace_name, cache.momentum_axis, key)
        else:
            x, y = compute_powder_line(workspace_name, cache.momentum_axis, key, cif_file=cif)
        if 'meV' not in cache.energy_axis.e_unit:
            from mslice.models.units import EnergyUnits
            import numpy as np
            y = np.array(y) * EnergyUnits(cache.energy_axis.e_unit).factor_from_meV()
        cache.overplot_lines[key] = plot_overplot_line(x, y, key, recoil, cache)

    def validate_intensity(self, intensity_start, intensity_end):
        intensity_start = self._to_float(intensity_start)
        intensity_end = self._to_float(intensity_end)
        if intensity_start is not None and intensity_end is not None and intensity_start > intensity_end:
            raise ValueError()
        return intensity_start, intensity_end

    def add_sample_temperature_field(self, field_name):
        self._sample_temp_fields.append(field_name)

    def update_sample_temperature(self, workspace_name):
        temp = sample_temperature(workspace_name, self._sample_temp_fields)
        self.set_sample_temperature(workspace_name, temp)

    def set_sample_temperature(self, workspace_name, temp):
        self._slice_cache[workspace_name].sample_temp = temp

    def workspace_selection_changed(self):
        pass
