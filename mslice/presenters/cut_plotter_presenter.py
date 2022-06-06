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


class CutPlotterPresenter(PresenterUtility):

    def __init__(self):
        self._main_presenter = None
        self._interactive_cut_cache = None
        self._cut_cache_dict = {}  # Dict of list of currently displayed cuts index by axes
        self._overplot_cache = {}

    def run_cut(self, workspace, cut, plot_over=False, save_only=False):
        workspace = get_workspace_handle(workspace)
        cut.workspace_name = workspace.name
        if save_only:
            self.save_cut_to_workspace(workspace, cut)
            return
        if cut.width is not None:
            self._plot_with_width(workspace, cut, plot_over)
        else:
            self._plot_cut(workspace, cut, plot_over)

    def _plot_cut(self, workspace, cut, plot_over, store=True, update_main=True):
        cut_axis = cut.cut_axis
        integration_axis = cut.integration_axis
        cut_ws = compute_cut(workspace, cut_axis, integration_axis, cut.norm_to_one, cut.algorithm, store)
        legend = generate_legend(workspace.name, integration_axis.units, integration_axis.start, integration_axis.end)
        en_conversion = self._main_presenter.is_energy_conversion_allowed() if self._main_presenter else True
        plot_cut_impl(cut_ws, (cut.intensity_start, cut.intensity_end), plot_over, legend, en_conversion)
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
            # The first plot will respect which button the user pressed. The rest will over plot
            plot_over = True
        cut.reset_integration_axis(cut.start, cut.end)

    def save_cache(self, ax, cut, plot_over=False):
        # If plot over is True you want to save all plotted cuts for use by the cli
        if ax not in self._cut_cache_dict.keys():
            self._cut_cache_dict[ax] = []
        if len(self._cut_cache_dict[ax]) == 0 or plot_over:
            self._cut_cache_dict[ax].append(cut)
        if not plot_over:
            self._cut_cache_dict[ax][:] = []
            self._cut_cache_dict[ax].append(cut)

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

    def plot_interactive_cut(self, workspace, cut, store):
        workspace = get_workspace_handle(workspace)
        self._plot_cut(workspace, cut, False, store, update_main=False)
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
            total_steps = np.log10(y2/y1)
        elif y1 < 0:
            y1_int = -1
            y2_int = 1
            total_steps = np.log10(y2 / y2_int) + np.log10(y1 / y1_int) + 2
        else:
            y1 = 1 if y1 == 0 else y1
            y2 = 1 if y2 == 0 else y2
            total_steps = np.log10(y2/y1) + 1

        adj_factor = total_steps * portion_of_axes / 2
        return np.resize(np.array([10**adj_factor, 10**(-adj_factor), np.nan]), size) * datum

    def add_overplot_line(self, workspace_name, key, recoil, cif=None, y_has_logarithmic=None, datum=None):
        datum = 0 if datum is None else datum
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
                y = self._get_log_bragg_y_coords(len(y), 0.15, datum)

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
