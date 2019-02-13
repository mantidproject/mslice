from mslice.views.cut_plotter import plot_cut_impl, draw_interactive_cut, cut_figure_exists
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.cut.cut import Cut
from mslice.models.labels import generate_legend
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
import mslice.plotting.pyplot as plt
from mslice.presenters.presenter_utility import PresenterUtility


class CutPlotterPresenter(PresenterUtility):

    def __init__(self):
        self._main_presenter = None
        self._cut_cache = {}
        self._cut_cache_list = []  # List of all currently displayed cuts created with plot_over set to True

    def run_cut(self, workspace, cut, plot_over=False, save_only=False):
        workspace = get_workspace_handle(workspace)
        cut.workspace_name = workspace.name
        self._cut_cache[workspace.name] = cut

        # If plot over is True you want to save all plotted cuts for use by the cli
        if len(self._cut_cache_list) == 0 or plot_over:
            self._cut_cache_list.append(cut)
        if not plot_over:
            self._cut_cache_list[:] = []
            self._cut_cache_list.append(cut)

        if cut.width is not None:
            self._plot_with_width(workspace, cut, plot_over)
        elif save_only:
            self.save_cut_to_workspace(workspace, cut)
        else:
            self._plot_cut(workspace, cut, plot_over)

    # def more_than_one_workspace_used_in_plot(self):
    #     last_cut = self._cut_cache_list[0]
    #     for cut in self._cut_cache_list[1:]:
    #         if last_cut.workspace_name != cut.workspace_name:
    #             return True
    #     return False

    def _plot_cut(self, workspace, cut, plot_over, store=True, update_main=True):
        cut_axis = cut.cut_axis
        integration_axis = cut.integration_axis
        cut_ws = compute_cut(workspace, cut_axis, integration_axis, cut.norm_to_one, store)
        legend = generate_legend(workspace.name, integration_axis.units, integration_axis.start,
                                 integration_axis.end)
        plot_cut_impl(cut_ws, cut_axis.units, (cut.intensity_start, cut.intensity_end), plot_over, legend)
        if update_main:
            self.set_is_icut(workspace.name, False)
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

    def save_cut_to_workspace(self, workspace, cut):
        compute_cut(workspace, cut.cut_axis, cut.integration_axis, cut.norm_to_one)
        self._main_presenter.update_displayed_workspaces()

    def plot_cut_from_selected_workspace(self, plot_over=False):
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        for workspace_name in selected_workspaces:
            self.plot_cut_from_workspace(workspace_name, plot_over)
            plot_over = True  # plot over if multiple workspaces selected

    def plot_cut_from_workspace(self, workspace, intensity_range=None, plot_over=False):

        workspace = get_workspace_handle(workspace)
        lines = plot_cut_impl(workspace, workspace.raw_ws.getDimension(0).getUnits(),
                              intensity_range=intensity_range, plot_over=plot_over)
        return lines

    def plot_interactive_cut(self, workspace, cut_axis, integration_axis, store):
        workspace = get_workspace_handle(workspace)
        cut = Cut(cut_axis, integration_axis, None, None)
        self._cut_cache[workspace.name] = cut
        self._plot_cut(workspace, cut, False, store, update_main=False)
        draw_interactive_cut(workspace)

    def store_icut(self, workspace_name, icut):
        self.set_is_icut(workspace_name, True)
        self._cut_cache[workspace_name].icut = icut

    def set_is_icut(self, workspace_name, is_icut):
        if not is_icut and workspace_name in self._cut_cache:
            self._cut_cache[workspace_name].icut = None
        if cut_figure_exists():
            plt.gcf().canvas.manager.is_icut(is_icut)

    def get_icut(self, workspace_name):
        try:
            return self._cut_cache[workspace_name].icut
        except KeyError:
            return None

    def update_main_window(self):
        if self._main_presenter is not None:
            self._main_presenter.highlight_ws_tab(2)
            self._main_presenter.update_displayed_workspaces()

    def workspace_selection_changed(self):
        pass
