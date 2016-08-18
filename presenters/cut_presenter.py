from views.cut_view import CutView
from widgets.cut.command import Command
# TODO move this class to presenters.axis
from slice_plotter_presenter import Axis


class CutPresenter(object):
    def __init__(self, cut_view, main_view, cut_plotter):
        self._cut_view = cut_view
        self._cut_plotter = cut_plotter
        self._main_presenter = main_view.get_presenter()
        # TODO wait for broadcast system to then call main_presenter.get_main_presenter().subscribe(self)
        if not isinstance(cut_view, CutView):
            raise TypeError("cut_view is not of type cut_view")
        # This should be populated according to the workspace
        self._cut_view.populate_cut_axis_options(['|Q|','DeltaE'])

    def notify(self, command):
        if command == Command.Plot:
            self._plot_cut(plot_over=False)
        elif command == Command.PlotOver:
            self._plot_cut(plot_over=True)
        elif command == Command.SaveToWorkspace:
            self._save_cut_to_workspace()
        elif command == Command.SaveToAscii:
            raise NotImplementedError('Save to ascii Not implemented')

    def _plot_cut(self, plot_over=False):
        cut_axis = Axis(self._cut_view.get_cut_axis(), self._cut_view.get_cut_axis_start(),
                        self._cut_view.get_cut_axis_step(), self._cut_view.get_cut_axis_end())
        integration_start = self._cut_view.get_integration_start()
        integration_end = self._cut_view.get_integration_end()
        intensity_start = self._cut_view.get_intensity_start()
        intensity_end = self._cut_view.get_intensity_end()
        norm_to_one = self._cut_view.get_intensity_is_norm_to_one()
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        if not selected_workspaces:
            raise NotImplementedError("Show an error message")
        if len(selected_workspaces)>1:
            raise NotImplementedError("Select only one workspace for now")
        selected_workspace = selected_workspaces[0]
        error_code = self._cut_plotter.plot_cut(selected_workspace, cut_axis, integration_start, integration_end,
                                   intensity_start, intensity_end, norm_to_one, plot_over)
        self._main_presenter.update__displayed_workspaces()
        # TODO handle error codes

    def _save_cut_to_workspace(self, plot_over=False):
        cut_axis = Axis(self._cut_view.get_cut_axis(), self._cut_view.get_cut_axis_start(),
                        self._cut_view.get_cut_axis_step(), self._cut_view.get_cut_axis_end())
        integration_start = self._cut_view.get_integration_start()
        integration_end = self._cut_view.get_integration_end()
        intensity_start = self._cut_view.get_intensity_start()
        intensity_end = self._cut_view.get_intensity_end()
        norm_to_one = self._cut_view.get_intensity_is_norm_to_one()
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        if not selected_workspaces:
            raise NotImplementedError("Show an error message")
        if len(selected_workspaces)>1:
            raise NotImplementedError("Select only one workspace for now")
        selected_workspace = selected_workspaces[0]
        error_code = self._cut_plotter.save_cut_to_workpsace(selected_workspace, cut_axis, integration_start,
                                                integration_end, intensity_start, intensity_end, norm_to_one, plot_over)
        self._main_presenter.update__displayed_workspaces()
        # TODO handle error codes
