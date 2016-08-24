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
        intensity_start, intensity_end, is_norm = params[4:]
        x,y = self._cut_algorithm.compute_cut(*params[:4])
        if is_norm:
            y = self._cut_algorithm.norm(y)
        self._plotting_module.plot(x, y, hold=plot_over)
        if intensity_start is None and intensity_end is None:
            self._plotting_module.autoscale()
        else:
            self._plotting_module.ylim(intensity_start, intensity_end)


    def _save_cut_to_workspace(self):
        try:
            params = self._parse_input()
        except ValueError:
            return
        cut_params = params[:4]
        x,y = self._cut_algorithm.compute_cut(*params[:4], keepworkspace=True)
        print ("workspaces will not be normalized , regardless of checkbox")
        self._main_presenter.update_displayed_workspaces()

    def _parse_input(self):
        # The messages of the raised exceptions are discarded. They are there for the sake of clarity/debugging
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        if len(selected_workspaces) != 1:
            self._cut_view.error_select_a_workspace()
            raise ValueError("Invalid workspace selection")
        selected_workspace = selected_workspaces[0]
        cut_axis = Axis(self._cut_view.get_cut_axis(), self._cut_view.get_cut_axis_start(),
                        self._cut_view.get_cut_axis_end(), self._cut_view.get_cut_axis_step())
        if cut_axis.units == "":
            self._cut_view.error_current_selection_invalid()
            raise ValueError("Not supported")
        try:
            cut_axis.start = float(cut_axis.start)
            cut_axis.end = float(cut_axis.end)
            cut_axis.step = float(cut_axis.step)
        except ValueError:
            self._cut_view.error_invalid_cut_axis_parameters()
            raise ValueError("Invalid Cut axis parameters")

        if None not in (cut_axis.start,cut_axis.end) and cut_axis.start > cut_axis.end:
            self._cut_view.error_invalid_cut_axis_parameters()
            raise ValueError("Invalid cut axis parameters")

        integration_start = self._cut_view.get_integration_start()
        integration_end = self._cut_view.get_integration_end()
        try:
            integration_start = float(integration_start)
            integration_end = float(integration_end)
        except ValueError:
            self._cut_view.error_invalid_integration_parameters()
            raise ValueError("Invalid integration parameters")

        if None not in (integration_start, integration_end) and integration_start >= integration_end:
            self._cut_view.error_invalid_integration_parameters()
            raise ValueError("Integration start >= Integration End")

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


    def workspace_selection_changed(self):
        workspace_selection = self._main_presenter.get_selected_workspaces()
        if len(workspace_selection) != 1:
            self._cut_view.clear_input_fields()
            return
        workspace_selection = workspace_selection[0]

        axis = self._cut_algorithm.get_available_axis(workspace_selection)
        if len(axis) != 2:
            self._cut_view.clear_input_fields()
            return
        self._cut_view.populate_cut_axis_options(axis)

