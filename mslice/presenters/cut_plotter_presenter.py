from mslice.models.cut.matplotlib_cut_plotter import plot_cut_impl
from mslice.models.cut.cut_functions import compute_cut
from mslice.presenters.presenter_utility import PresenterUtility

class CutPlotterPresenter(PresenterUtility):

    def __init__(self):
        self.main_presenter = None
        self._cut_cache = {}

    def run_cut(self, workspace, cut_cache, plot_over=False, save_only=False):
        workspace = get_workspace_handle(workspace)
        self._cut_cache[workspace.name] = cut_cache
        if cut_cache.width is not None:
            self._plot_with_width(workspace, cut_cache, plot_over)
        elif save_only:
            self.save_cut_to_workspace(workspace, cut_cache)
        else:
            self._plot_cut(workspace, params, plot_over)

    def plot_cut(self, workspace, cut_cache, plot_over, store=True, update_main=True):
        cut_axis = cut_cache.cut_axis
        integration_axis = cut_cache.integration_axis
        cut = compute_cut(workspace, cut_axis, integration_axis, cut_cache.norm_to_one, store)
        legend = generate_legend(workspace.name, integration_axis.units, integration_axis.start,
                                 integration_axis.end)
        plot_cut_impl(cut, cut_axis.units, (cut_cache.intensity_start, cut_cache.intensity_end), plot_over, legend)
        if update_main:
            self.set_is_icut(workspace.name, False)
            self._main_presenter.highlight_ws_tab(2)
            self._main_presenter.update_displayed_workspaces()

    def _plot_with_width(self, workspace, cut_cache, plot_over):
        """This function handles the width parameter."""
        integration_start = cut_cache.integration_axis.start
        integration_end = cut_cache.integration_axis.end
        cut_start, cut_end = integration_start, min(integration_start + width, integration_end)
        while cut_start != cut_end:
            cut_cache.integration_axis.start = cut_start
            cut_cache.integration_axis.end = cut_end
            self.plot_cut(workspace, cut_cache, plot_over)
            cut_start, cut_end = cut_end, min(cut_end + width, integration_end)
            # The first plot will respect which button the user pressed. The rest will over plot
            plot_over = True

    def save_cut_to_workspace(self, workspace, cut_cache):
        compute_cut(workspace, cut_cache.cut_axis, cut_cache.integration_axis, cut_cache.norm_to_one)
        self._main_presenter.update_displayed_workspaces()

    def plot_interactive_cut(self, workspace, cut_axis, integration_axis, store):
        cache = CutCache(cut_axis, integration_axis, None, None)
        self.plot_cut(workspace, cache, False, store, update_main=False)
        draw_interactive_cut(workspace)

    def store_icut(self, workspace_name, icut):
        self.set_is_icut(True, workspace_name)
        self._slice_cache[workspace_name].icut = icut

    def set_is_icut(self, workspace_name, is_icut):
        if not is_icut:
            self._slice_cache[workspace_name].icut = None
        plt.gcf().canvas.manager.is_icut(is_icut)

    def get_icut(self, workspace_name):
        return self._slice_cache[workspace_name].icut