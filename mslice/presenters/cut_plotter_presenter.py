import numpy as np

from mslice.views.cut_plotter import plot_cut_impl, draw_interactive_cut, cut_figure_exists
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.labels import generate_legend, is_momentum, is_twotheta
from mslice.models.workspacemanager.workspace_algorithms import export_workspace_to_ads
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
import mslice.plotting.pyplot as plt
from mslice.presenters.presenter_utility import PresenterUtility
from mslice.plotting.plot_window.overplot_interface import remove_line, plot_overplot_line
from mslice.models.powder.powder_functions import compute_powder_line
import warnings

BRAGG_SIZE_ON_AXES = 0.15


class CutPlotterPresenter(PresenterUtility):

    def __init__(self):
        self._main_presenter = None
        self._interactive_cut_cache = None
        self._cut_cache_dict = {}  # Dict of list of currently displayed cuts index by axes
        self._temp_cut_cache = []
        self._overplot_cache = {}

    def run_cut(self, workspace, cut, plot_over=False, save_only=False):
        workspace = get_workspace_handle(workspace)
        if save_only:
            self.save_cut_to_workspace(workspace, cut)
            return
        if cut.width is not None:
            self._plot_with_width(workspace, cut, plot_over)
        else:
            self._plot_cut(workspace, cut, plot_over)

    def _plot_cut(self, workspace, cut, plot_over, store=True, update_main=True,
                  intensity_correction="scattering_function"):
        cut_axis = cut.cut_axis
        integration_axis = cut.integration_axis
        if not cut.cut_ws:
            cut.cut_ws = compute_cut(workspace, cut_axis, integration_axis, cut.norm_to_one, cut.algorithm, store)
        if intensity_correction == "scattering_function":
            cut_ws = cut.cut_ws
            intensity_range = (cut.intensity_start, cut.intensity_end)
        else:
            cut_ws = cut.get_intensity_corrected_ws(intensity_correction)
            intensity_range = cut.get_corrected_intensity_range(intensity_correction)
        legend = generate_legend(workspace.name, integration_axis.units, integration_axis.start, integration_axis.end)
        en_conversion = self._main_presenter.is_energy_conversion_allowed() if self._main_presenter else True
        plot_cut_impl(cut_ws, intensity_range, plot_over, legend, en_conversion)
        if update_main:
            self.set_is_icut(False)
            self.update_main_window()

    def _plot_with_width(self, workspace, cut, plot_over):
        """This function handles the width parameter."""
        integration_start = cut.integration_axis.start
        integration_end = cut.integration_axis.end
        cut_start, cut_end = integration_start, min(integration_start + cut.width, integration_end)
        while cut_start != cut_end:
            cut.integration_axis.start = cut_start
            cut.integration_axis.end = cut_end
            self._plot_cut(workspace, cut, plot_over)
            cut_start, cut_end = cut_end, min(cut_end + cut.width, integration_end)
            cut.cut_ws = None
            # The first plot will respect which button the user pressed. The rest will over plot
            plot_over = True
        cut.reset_integration_axis(cut.start, cut.end)

    def save_cache(self, ax, cut, plot_over=False):
        # If plot over is True you want to save all plotted cuts for use by the cli
        if cut.intensity_corrected:
            return
        if ax not in self._cut_cache_dict.keys():
            self._cut_cache_dict[ax] = []
        if len(self._cut_cache_dict[ax]) == 0:
            self._cut_cache_dict[ax].append(cut)
        else:
            cut_already_cached, cached_cut = self._cut_already_cached(cut)
            if not cut_already_cached:
                if not plot_over:
                    self._cut_cache_dict[ax] = []
                self._cut_cache_dict[ax].append(cut)
            else:
                if not plot_over:
                    self._cut_cache_dict[ax] = [cached_cut]
                elif cut not in self._cut_cache_dict[ax]:
                    self._cut_cache_dict[ax].append(cached_cut)

    def _cut_already_cached(self, cut):
        for cached_cut in self._temp_cut_cache:
            if self._cut_is_equal(cached_cut, cut):
                return True, cached_cut
        return False, None

    @staticmethod
    def _cut_is_equal(cached_cut, cut):
        cached_cut_params = [cached_cut.cut_axis, cached_cut.integration_axis, cached_cut.intensity_start,
                             cached_cut.intensity_end, cached_cut.norm_to_one, cached_cut.width, cached_cut.algorithm,
                             cached_cut.cut_ws]
        cut_params = [cut.cut_axis, cut.integration_axis, cut.intensity_start, cut.intensity_end,
                      cut.norm_to_one, cut.width, cut.algorithm, cut.cut_ws]
        if cached_cut_params == cut_params:
            return True
        else:
            return False

    def remove_cut_from_cache_by_index(self, ax, index):
        del self._cut_cache_dict[ax][index]

    def get_cache(self, ax):
        return self._cut_cache_dict[ax] if ax in self._cut_cache_dict.keys() else None

    def save_cut_to_workspace(self, workspace, cut):
        cut_ws = compute_cut(workspace, cut.cut_axis, cut.integration_axis, cut.norm_to_one, cut.algorithm)
        self._main_presenter.update_displayed_workspaces()
        export_workspace_to_ads(cut_ws)

    def plot_cut_from_selected_workspace(self, plot_over=False):
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        for workspace_name in selected_workspaces:
            self.plot_cut_from_workspace(workspace_name, plot_over)
            plot_over = True  # plot over if multiple workspaces selected

    def plot_cut_from_workspace(self, workspace, plot_over=False, intensity_range=None):
        workspace = get_workspace_handle(workspace)
        lines = plot_cut_impl(workspace, intensity_range=intensity_range, plot_over=plot_over)
        self.set_is_icut(False)
        return lines

    def plot_interactive_cut(self, workspace, cut, store, intensity_correction):
        self._plot_cut(workspace, cut, False, store, update_main=False, intensity_correction=intensity_correction)
        draw_interactive_cut(workspace)

    def hide_overplot_line(self, workspace, key):
        cache = self._overplot_cache
        if key in cache:
            line = cache.pop(key)
            remove_line(line)

    @staticmethod
    def _get_log_bragg_y_coords(size, portion_of_axes, datum):
        datum = 0.001 if datum == 0 else datum
        y1, y2 = plt.gca().get_ylim()
        if (y2 > 0 and y1 > 0) or (y2 < 0 and y1 < 0):
            total_steps = np.log10(y2 / y1)
        elif y1 < 0:
            total_steps_up = np.log10(y2) + 1 if abs(y2) >= 1 else abs(y2)
            total_steps_down = np.log10(-y1) + 1 if abs(y1) >= 1 else abs(y1)
            total_steps = total_steps_up + total_steps_down
        else:
            y1 = 1 if y1 == 0 else y1
            y2 = 1 if y2 == 0 else y2
            total_steps = np.log10(y2 / y1) + 1

        adj_factor = total_steps * portion_of_axes / 2
        return np.resize(np.array([10 ** adj_factor, 10 ** (-adj_factor), np.nan]), size) * datum

    def add_overplot_line(self, workspace_name, key, recoil, cif=None, y_has_logarithmic=None, datum=0):
        cache = self._cut_cache_dict[plt.gca()][0]
        cache.rotated = not is_twotheta(cache.cut_axis.units) and not is_momentum(cache.cut_axis.units)
        try:
            ws_handle = get_workspace_handle(workspace_name)
            workspace_name = ws_handle.parent
            scale_fac = np.nanmax(ws_handle.get_signal()) / 10
        except KeyError:
            # Workspace is interactively generated and is not in the workspace list
            scale_fac = 90
            workspace_name = workspace_name.split('(')[0][:-4]
        if cache.rotated:
            q_axis = cache.integration_axis
        else:
            q_axis = cache.cut_axis
        x, y = compute_powder_line(workspace_name, q_axis, key, cif_file=cif)
        try:
            if not y_has_logarithmic:
                y = np.array(y) * scale_fac / np.nanmax(y) + datum
            else:
                y = self._get_log_bragg_y_coords(len(y), BRAGG_SIZE_ON_AXES, datum)

            self._overplot_cache[key] = plot_overplot_line(x, y, key, recoil, cache)
        except (ValueError, IndexError):
            warnings.warn("No Bragg peak found.")

    def store_icut(self, icut):
        self._interactive_cut_cache = icut

    def set_is_icut(self, is_icut):
        if cut_figure_exists():
            plt.gcf().canvas.manager.set_is_icut(is_icut)

    def get_icut(self):
        return self._interactive_cut_cache

    def update_main_window(self):
        if self._main_presenter is not None:
            self._main_presenter.highlight_ws_tab(2)
            self._main_presenter.update_displayed_workspaces()

    def workspace_selection_changed(self):
        pass

    def is_overplot(self, line):
        return line in self._overplot_cache.values()

    def _show_intensity(self, cut_cache, intensity_correction):
        plot_over = False
        self._temp_cut_cache = list(cut_cache)
        for cached_cut in self._temp_cut_cache:
            workspace = get_workspace_handle(cached_cut.workspace_name)
            self._plot_cut(workspace, cached_cut, plot_over=plot_over, intensity_correction=intensity_correction)
            plot_over = True
        self._temp_cut_cache = []

    def show_scattering_function(self, axes):
        self._show_intensity(self._cut_cache_dict[axes], "scattering_function")

    def show_dynamical_susceptibility(self, axes):
        self._show_intensity(self._cut_cache_dict[axes], "dynamical_susceptibility")

    def show_dynamical_susceptibility_magnetic(self, axes):
        self._show_intensity(self._cut_cache_dict[axes], "dynamical_susceptibility_magnetic")

    def show_d2sigma(self, axes):
        self._show_intensity(self._cut_cache_dict[axes], "d2sigma")

    def show_symmetrised(self, axes):
        self._show_intensity(self._cut_cache_dict[axes], "symmetrised")

    def set_sample_temperature(self, axes, ws_name, temp):
        cut_dict = {}
        parent_ws_name = None
        for cut in self._cut_cache_dict[axes]:
            if ws_name == cut.workspace_name:
                parent_ws_name = cut.parent_ws_name
            if cut.parent_ws_name not in cut_dict.keys():
                cut_dict[cut.parent_ws_name] = [cut]
            else:
                cut_dict[cut.parent_ws_name].append(cut)

        if parent_ws_name:
            for cut in cut_dict[parent_ws_name]:
                cut.sample_temp = temp
        else:
            warnings.warn("Sample temperature not set, cut not found in cache")

    def propagate_sample_temperatures_throughout_cache(self, axes):
        if len(self._cut_cache_dict[axes]) <= 1:
            return False

        temperature_dict = {}
        cuts_with_no_temp = []
        for cut in self._cut_cache_dict[axes]:
            if cut.raw_sample_temp and cut.parent_ws_name not in temperature_dict.keys():
                temperature_dict[cut.parent_ws_name] = cut.sample_temp
            elif not cut.raw_sample_temp:
                cuts_with_no_temp.append(cut)

        if len(temperature_dict) > 0:
            for cut in list(cuts_with_no_temp):
                if cut.parent_ws_name in temperature_dict.keys():
                    cut.sample_temp = temperature_dict[cut.parent_ws_name]
                    cuts_with_no_temp.remove(cut)

        if len(cuts_with_no_temp) == 0:
            return True
        else:
            return False
