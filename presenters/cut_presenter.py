from views.cut_view import CutView
from widgets.cut.command import Command
from validation_decorators import require_main_presenter

class Axis(object):
    def __init__(self, units,start, end, step):
        self.start = start
        self.end = end
        self.step = step
        self.units =units

class CutPresenter(object):
    def __init__(self, cut_view, cut_algorithm, plotting_module):
        self._cut_view = cut_view
        self._cut_algorithm = cut_algorithm
        if not isinstance(cut_view, CutView):
            raise TypeError("cut_view is not of type cut_view")
        self._main_presenter = None
        self._plotting_module = plotting_module

    def register_master(self, main_presenter):
        self._main_presenter = main_presenter
        self._main_presenter.subscribe_to_workspace_selection_monitor(self)

    def workspace_selection_changed(self):
        pass

    @require_main_presenter
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
        try:
            params = self._parse_input()
        except ValueError:
            return
        cut_params = params[:4]
        intensity_start ,intensity_end, is_norm = params[4:]
        x,y = self._cut_algorithm.compute_cut(*params[:4])
        self._plotting_module.plot(x,y)
        self._main_presenter.update_displayed_workspaces()

    def _save_cut_to_workspace(self, plot_over=False):
        raise NotImplementedError('Not implemented')

    def _parse_input(self):
        # The messages of the raised exceptions are discarded. They are there for the sake of clarity/debugging
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        if len(selected_workspaces) != 1:
            self._cut_view.error_select_workspace()
            raise ValueError("Invalid workspace selection")
        selected_workspace = selected_workspaces[0]
        cut_axis = Axis(self._cut_view.get_cut_axis(), self._cut_view.get_cut_axis_start(),
                        self._cut_view.get_cut_axis_end(), self._cut_view.get_cut_axis_step())
        try:
            cut_axis.start = self._to_float(cut_axis.start)
            cut_axis.end = self._to_float(cut_axis.end)
            cut_axis.step = self._to_float(cut_axis.step)
        except ValueError:
            self._cut_view.error_invalid_cut_axis_parameters()
            raise ValueError("Invalid Cut axis parameters")

        if None not in (cut_axis.start,cut_axis.end) and cut_axis.start > cut_axis.end:
            self._cut_view.error_invalid_cut_axis_parameters()
            raise ValueError("Invalid cut axis parameters")

        integration_start = self._cut_view.get_integration_start()
        integration_end = self._cut_view.get_integration_end()
        try:
            integration_start = self._to_float(integration_start)
            integration_end = self._to_float(integration_end)
        except ValueError:
            self._cut_view.error_invalid_integration_parameters()
            raise ValueError("Invalid integration parameters")

        if None not in (integration_start, integration_end) and integration_start > integration_end:
            self._cut_view.error_invalid_integration_parameters()
            raise ValueError("Integration start > Integration End")

        intensity_start = self._cut_view.get_intensity_start()
        intensity_end = self._cut_view.get_intensity_end()
        try:
            intensity_start = self._to_float(intensity_start)
            intensity_end = self._to_float(intensity_end)
        except ValueError:
            self._cut_view.error_invalid_intensity_parameters()
            raise ValueError("Invalid intensity params")

        norm_to_one = bool(self._cut_view.get_intensity_is_norm_to_one())
        return selected_workspace, cut_axis, integration_start, integration_end, intensity_start, intensity_end, norm_to_one

    def _to_float(self, x):
        if x == "":
            return None
        return float(x)