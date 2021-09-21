from mslice.views.cut_plotter import plot_cut_impl, draw_interactive_cut, cut_figure_exists
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.labels import generate_legend, is_momentum, is_twotheta
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

        if cut.width is not None:
            self._plot_with_width(workspace, cut, plot_over)
        elif save_only:
            self.save_cut_to_workspace(workspace, cut)
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
        compute_cut(workspace, cut.cut_axis, cut.integration_axis, cut.norm_to_one, cut.algorithm)
        self._main_presenter.update_displayed_workspaces()

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

    def add_overplot_line(self, workspace_name, key, recoil, cif=None):
        recoil = False
        from mslice.plotting.pyplot import gca
        cache = self._cut_cache_dict[gca()][0]
        cache.rotated = not is_twotheta(cache.cut_axis.units) and not is_momentum(cache.cut_axis.units)
        import numpy as np
        try:
            ws_handle = get_workspace_handle(workspace_name)
            workspace_name = ws_handle.parent
            scale_fac = np.nanmax(ws_handle.get_signal()) / 10
        except KeyError:
            # Workspace is interactively generated and is not in the workspace list
            scale_fac = 1
            workspace_name = workspace_name.split('(')[0][:-4]
        if cache.rotated:
            q_axis = cache.integration_axis
        else:
            q_axis = cache.cut_axis
        x, y = compute_powder_line(workspace_name, q_axis, key, cif_file=cif)
        try:
            y = np.array(y) * (scale_fac / np.nanmax(y))
            self._overplot_cache[key] = plot_overplot_line(x, y, key, recoil, cache)
        except ValueError:
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
