import numpy as np

from mslice.views.cut_plotter import (
    plot_cut_impl,
    draw_interactive_cut,
    cut_figure_exists,
    get_current_plot,
)
from mslice.models.alg_workspace_ops import get_range_end
from mslice.models.cut.cut import SampleTempValueError
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.labels import generate_legend
from mslice.models.workspacemanager.workspace_algorithms import export_workspace_to_ads
from mslice.models.workspacemanager.workspace_provider import (
    add_workspace,
    get_workspace_handle,
)
import mslice.plotting.pyplot as plt
from mslice.presenters.presenter_utility import PresenterUtility
from mslice.plotting.plot_window.overplot_interface import (
    remove_line,
    plot_overplot_line,
)
from mslice.models.powder.powder_functions import compute_powder_line
from mslice.models.intensity_correction_algs import sample_temperature
from mslice.models.axis import Axis
from mslice.util.intensity_correction import IntensityType, IntensityCache
from mslice.util.mantid.algorithm_wrapper import remove_from_ads
import warnings
from sys import float_info

BRAGG_SIZE_ON_AXES = 0.15


class CutPlotterPresenter(PresenterUtility):
    _current_icut = None  # static variable, as only one icut can be open at any time.
    _prepared_cut_for_cache = (
        None  # static variable, used and reset to None upon plotting of a cut
    )
    # needs to be static as icut design means that icut does not have access to correct CPP on first plot

    def __init__(self):
        self._main_presenter = None
        self._interactive_cut_cache = None
        self._interactive_prev = None
        self._cut_cache_dict = {}  # Dict of list of currently displayed cuts index by axes
        self._temp_cut_cache = []
        self._overplot_cache = {}
        self._cache_intensity_correction_methods()

    def run_cut(self, workspace, cut, plot_over=False, save_only=False):
        workspace = get_workspace_handle(workspace)
        if save_only:
            self.save_cut_to_workspace(workspace, cut)
            return
        if cut.width is not None:
            return self._plot_with_width(workspace, cut, plot_over)
        else:
            self._plot_cut(workspace, cut, plot_over)

    def _plot_cut(
        self,
        workspace,
        cut,
        plot_over,
        store=True,
        update_main=True,
        intensity_correction=IntensityType.SCATTERING_FUNCTION,
        final_plot=True,
    ):
        cut_axis = cut.cut_axis
        integration_axis = cut.integration_axis
        if not cut.cut_ws:
            cut.cut_ws = compute_cut(
                workspace,
                cut_axis,
                integration_axis,
                cut.norm_to_one,
                cut.algorithm,
                store,
            )
            self.prepare_cut_for_cache(cut)
        if intensity_correction == IntensityType.SCATTERING_FUNCTION:
            cut_ws = cut.cut_ws
            intensity_range = (cut.intensity_start, cut.intensity_end)
        else:
            cut_ws = cut.get_intensity_corrected_ws(intensity_correction)
            intensity_range = cut.get_corrected_intensity_range(intensity_correction)
        ws_label_name = workspace.parent if workspace.parent else workspace.name
        legend = generate_legend(
            ws_label_name,
            integration_axis.units,
            integration_axis.start,
            integration_axis.end,
        )
        en_conversion = (
            self._main_presenter.is_energy_conversion_allowed()
            if self._main_presenter
            else True
        )
        plot_cut_impl(cut_ws, intensity_range, plot_over, legend, en_conversion)
        current_plot_intensity = self.get_current_plot_intensity()
        if (
            final_plot
            and plot_over
            and current_plot_intensity
            and not intensity_correction == current_plot_intensity
        ):
            self.apply_intensity_correction_after_plot_over(current_plot_intensity)
        if update_main:
            self.set_is_icut(False)
            self.update_main_window()

    def prepare_cut_for_cache(self, cut):
        CutPlotterPresenter._prepared_cut_for_cache = cut.copy_for_cache()

    def cache_prepared_cut(self, ax, plot_over):
        self.save_cache(ax, CutPlotterPresenter._prepared_cut_for_cache, plot_over)
        CutPlotterPresenter._prepared_cut_for_cache = None

    def get_prepared_cut_for_cache(self):
        return CutPlotterPresenter._prepared_cut_for_cache

    def _plot_with_width(self, workspace, cut, plot_over):
        """This function handles the width parameter."""
        integration_start = cut.integration_axis.start
        integration_end = cut.integration_axis.end
        cut_start = integration_start
        cut_end = get_range_end(integration_start, integration_end, cut.width)
        while cut_start != cut_end:
            cut.integration_axis.start = cut_start
            cut.integration_axis.end = cut_end
            final_plot = True if cut_start + cut.width == integration_end else False
            self._plot_cut(workspace, cut, plot_over, final_plot=final_plot)
            cut_start = cut_end
            cut_end = get_range_end(cut_end, integration_end, cut.width)
            cut.cut_ws = None
            # The first plot will respect which button the user pressed. The rest will over plot
            plot_over = True
        cut.reset_integration_axis(cut.start, cut.end)

        return cut.cut_axis.validate_step_against_workspace(workspace)

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
        cached_cut_params = [
            cached_cut.cut_axis,
            cached_cut.integration_axis,
            cached_cut.intensity_start,
            cached_cut.intensity_end,
            cached_cut.norm_to_one,
            cached_cut.width,
            cached_cut.algorithm,
            cached_cut.cut_ws,
        ]
        cut_params = [
            cut.cut_axis,
            cut.integration_axis,
            cut.intensity_start,
            cut.intensity_end,
            cut.norm_to_one,
            cut.width,
            cut.algorithm,
            cut.cut_ws,
        ]
        if cached_cut_params == cut_params:
            return True
        else:
            return False

    def remove_cut_from_cache_by_index(self, ax, index):
        del self._cut_cache_dict[ax][index]
        return len(self._cut_cache_dict[ax])

    def get_cache(self, ax):
        return self._cut_cache_dict[ax] if ax in self._cut_cache_dict.keys() else None

    def save_cut_to_workspace(self, workspace, cut):
        cut_ws = compute_cut(
            workspace,
            cut.cut_axis,
            cut.integration_axis,
            cut.norm_to_one,
            cut.algorithm,
        )
        self._main_presenter.update_displayed_workspaces()
        export_workspace_to_ads(cut_ws)

    def plot_cut_from_selected_workspace(self, plot_over=False):
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        for workspace_name in selected_workspaces:
            self.plot_cut_from_workspace(workspace_name, plot_over)
            plot_over = True  # plot over if multiple workspaces selected

    def plot_cut_from_workspace(
        self, workspace, plot_over=False, intensity_range=(None, None)
    ):
        workspace = get_workspace_handle(workspace)
        lines = plot_cut_impl(
            workspace, intensity_range=intensity_range, plot_over=plot_over
        )
        self.set_is_icut(False)
        return lines

    def plot_interactive_cut(self, workspace, cut, store, intensity_correction):
        self._plot_cut(
            workspace,
            cut,
            False,
            store,
            update_main=False,
            intensity_correction=intensity_correction,
        )
        if cut._cut_ws is not None:
            raw_name = cut._cut_ws.raw_ws.name()
            if self._interactive_prev is not None and self._interactive_prev != raw_name:
                remove_from_ads(self._interactive_prev)
            self._interactive_prev = raw_name
        draw_interactive_cut(workspace)
        self.set_icut_cut(cut)

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
        return (
            np.resize(np.array([10**adj_factor, 10 ** (-adj_factor), np.nan]), size)
            * datum
        )

    def add_overplot_line(
        self,
        workspace_name,
        key,
        recoil,
        cif=None,
        e_is_logarithmic=None,
        datum=0,
        intensity_correction=IntensityType.SCATTERING_FUNCTION,
        **kwargs,
    ):
        cache = self._cut_cache_dict[plt.gca()][0]
        if cache.rotated:
            warnings.warn("No Bragg peak found as cut has no |Q| dimension.")
            return
        try:
            ws_handle = get_workspace_handle(workspace_name)
            if not ws_handle.parent:
                workspace_name = workspace_name.split("(")[0][:-4]
            else:
                workspace_name = ws_handle.parent
        except KeyError:
            # Workspace is interactively generated and is not in the workspace list
            workspace_name = workspace_name.split("(")[0][:-4]

        # Get 10% of the maximum signal
        scale_fac = self._get_overall_max_signal(intensity_correction) / 10

        q_axis = self._get_overall_q_axis()
        x, y = compute_powder_line(workspace_name, q_axis, key, cif_file=cif)
        try:
            if not e_is_logarithmic:
                y = np.array(y) * scale_fac / np.nanmax(y) + datum
            else:
                y = self._get_log_bragg_y_coords(len(y), BRAGG_SIZE_ON_AXES, datum)

            self._overplot_cache[key] = plot_overplot_line(
                x, y, key, recoil, cache, **kwargs
            )
        except (ValueError, IndexError):
            warnings.warn("No Bragg peak found.")

    def _get_overall_q_axis(self):
        min_q = float_info.max
        max_q = -min_q
        for cut in self._cut_cache_dict[plt.gca()]:
            if cut.q_axis.end > max_q:
                max_q = cut.q_axis.end
            if cut.q_axis.start < min_q:
                min_q = cut.q_axis.start
        return Axis(cut.q_axis.units, min_q, max_q, cut.q_axis.step, cut.q_axis.e_unit)

    def _get_overall_max_signal(self, intensity_correction):
        overall_max_signal = 0
        for cut in self._cut_cache_dict[plt.gca()]:
            try:
                cut.sample_temp
            except SampleTempValueError:
                try:
                    self.propagate_sample_temperatures_throughout_cache(plt.gca())
                except RuntimeError:
                    continue
            ws = cut.get_intensity_corrected_ws(intensity_correction)
            max_cut_signal = np.nanmax(ws.get_signal())
            if max_cut_signal > overall_max_signal:
                overall_max_signal = max_cut_signal
        return overall_max_signal

    def set_is_icut(self, is_icut):
        if cut_figure_exists():
            plt.gcf().canvas.manager.set_is_icut(is_icut)

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
            self._plot_cut(
                workspace,
                cached_cut,
                plot_over=plot_over,
                intensity_correction=intensity_correction,
            )
            plot_over = True
        self._temp_cut_cache = []

    def show_scattering_function(self, axes):
        for key, value in self._cut_cache_dict.items():
            if key == axes:
                self._show_intensity(value, IntensityType.SCATTERING_FUNCTION)
                break

    def show_dynamical_susceptibility(self, axes):
        for key, value in self._cut_cache_dict.items():
            if key == axes:
                self._show_intensity(value, IntensityType.CHI)
                break

    def show_dynamical_susceptibility_magnetic(self, axes):
        for key, value in self._cut_cache_dict.items():
            if key == axes:
                self._show_intensity(value, IntensityType.CHI_MAGNETIC)
                break

    def show_d2sigma(self, axes):
        for key, value in self._cut_cache_dict.items():
            if key == axes:
                self._show_intensity(value, IntensityType.D2SIGMA)
                break

    def show_symmetrised(self, axes):
        for key, value in self._cut_cache_dict.items():
            if key == axes:
                self._show_intensity(value, IntensityType.SYMMETRISED)
                break

    def show_gdos(self, axes):
        for key, value in self._cut_cache_dict.items():
            if key == axes:
                self._show_intensity(value, IntensityType.GDOS)
                break

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
            if (
                cut.raw_sample_temp
                and cut.parent_ws_name not in temperature_dict.keys()
            ):
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

    def set_sample_temperature_by_field(self, axes, field, workspace_name):
        temp = sample_temperature(workspace_name, [field])
        self.set_sample_temperature(axes, workspace_name, temp)

    def get_icut(self):
        return self._interactive_cut_cache

    def store_icut(self, icut):
        self._interactive_cut_cache = icut

    @staticmethod
    def get_icut_cut():
        return CutPlotterPresenter._current_icut

    @staticmethod
    def store_icut_cut():
        cut = CutPlotterPresenter.get_icut_cut()
        if cut:
            add_workspace(cut.cut_ws, cut.workspace_name)

    @staticmethod
    def set_icut_cut(icut_cut):
        CutPlotterPresenter._current_icut = icut_cut

    @staticmethod
    def set_icut_intensity_category(intensity_type):
        icut_plot = get_current_plot()  # icut is locked to current
        if hasattr(icut_plot, "set_intensity_from_type"):
            icut_plot.set_intensity_from_type(intensity_type)

    @staticmethod
    def get_current_plot_intensity():
        return get_current_plot().intensity_type

    @staticmethod
    def apply_intensity_correction_after_plot_over(intensity_type):
        return get_current_plot().trigger_action_from_type(intensity_type)

    def _cache_intensity_correction_methods(self):
        cat = plt.CATEGORY_CUT
        IntensityCache.cache_method(
            cat,
            IntensityType.SCATTERING_FUNCTION,
            self.show_scattering_function.__name__,
        )
        IntensityCache.cache_method(
            cat, IntensityType.CHI, self.show_dynamical_susceptibility.__name__
        )
        IntensityCache.cache_method(
            cat,
            IntensityType.CHI_MAGNETIC,
            self.show_dynamical_susceptibility_magnetic.__name__,
        )
        IntensityCache.cache_method(
            cat, IntensityType.D2SIGMA, self.show_d2sigma.__name__
        )
        IntensityCache.cache_method(
            cat, IntensityType.SYMMETRISED, self.show_symmetrised.__name__
        )
        IntensityCache.cache_method(cat, IntensityType.GDOS, self.show_gdos.__name__)
