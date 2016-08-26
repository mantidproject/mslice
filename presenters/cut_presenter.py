from views.cut_view import CutView
from widgets.cut.command import Command
from validation_decorators import require_main_presenter
from presenters.slice_plotter_presenter import Axis
INTENSITY = 'Signal/#Events'


class CutPresenter(object):
    def __init__(self, cut_view, cut_algorithm, plotting_module):
        self._cut_view = cut_view
        self._cut_algorithm = cut_algorithm
        if not isinstance(cut_view, CutView):
            raise TypeError("cut_view is not of type CutView")
        self._main_presenter = None
        self._plotting_module = plotting_module
        self._acting_on = None

    def register_master(self, main_presenter):
        self._main_presenter = main_presenter
        self._main_presenter.subscribe_to_workspace_selection_monitor(self)

    @require_main_presenter
    def notify(self, command):
        if command == Command.Plot:
            self._process_cuts(plot_over=False)
        elif command == Command.PlotOver:
            self._process_cuts(plot_over=True)
        elif command == Command.SaveToWorkspace:
            self._save_cut_to_workspace()
        elif command == Command.SaveToAscii:
            raise NotImplementedError('Save to ascii Not implemented')

    def _process_cuts(self, plot_over):
        """This function handles the width parameter. If it is not specified a single cut is plotted from
        integration_start to integration_end """
        try:
            params = self._parse_input()
        except ValueError:
            return
        width = params[-1]
        params = params[:-1]
        if width is None:
            # No width specified, just plot a single cut
            self._plot_cut(params, plot_over)
            return
        integration_start, integration_end = params[2:4]
        cut_start, cut_end = integration_start, min(integration_start + width, integration_end)
        while cut_start != cut_end:
            params = params[:2] + (cut_start, cut_end) + params[4:]
            self._plot_cut(params, plot_over)
            cut_start, cut_end = cut_end, min(cut_end + width, integration_end)
            # The first plot will respect which button the user pressed. The rest will over plot
            plot_over = True

    def _plot_cut(self,params, plot_over):
        cut_params = params[:5]
        intensity_start, intensity_end, integration_axis = params[5:]
        x, y, e = self._cut_algorithm.compute_cut_xye(*cut_params)

        legend = self._get_legend(params[0], integration_axis, params[2:4])
        self._plotting_module.errorbar(x, y, yerr=e, label=legend, hold=plot_over)
        self._plotting_module.legend()
        self._plotting_module.xlabel(cut_params[1].units)
        self._plotting_module.ylabel(INTENSITY)

        if intensity_start is None and intensity_end is None:
            self._plotting_module.autoscale()
        else:
            self._plotting_module.ylim(intensity_start, intensity_end)

    def _save_cut_to_workspace(self):
        try:
            params = self._parse_input()
        except ValueError:
            return
        cut_params = params[:5]
        self._cut_algorithm.compute_cut(*cut_params)
        self._main_presenter.update_displayed_workspaces()

    def _get_legend(self, workspace_name, integrated_dim, integration_range):
        if integrated_dim == 'DeltaE':
            integrated_dim = 'e'
        return workspace_name + "    " + "%.2f"%integration_range[0] + "<"+integrated_dim+"<"+\
            "%.2f"%integration_range[1]

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

        if None not in (cut_axis.start,cut_axis.end) and cut_axis.start >= cut_axis.end:
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
        width = self._cut_view.get_integration_width()
        if width.strip():
            try:
                width = float(width)
            except ValueError:
                self._cut_view.error_invalid_width()
                raise ValueError("Invalid width")
        else:
            width = None
        integration_axis = self._cut_algorithm.get_other_axis(selected_workspace, cut_axis)
        return selected_workspace, cut_axis, integration_start, integration_end, norm_to_one, intensity_start, \
               intensity_end, integration_axis, width

    def _to_float(self, x):
        if x == "":
            return None
        return float(x)

    def workspace_selection_changed(self):
        self._cut_view.clear_input_fields()
        self._acting_on = None
        workspace_selection = self._main_presenter.get_selected_workspaces()

        if len(workspace_selection) != 1:
            self._cut_view.disable()
            return

        workspace = workspace_selection[0]
        if self._cut_algorithm.is_cuttable(workspace):
            self._acting_on = "cuttable"
            axis = self._cut_algorithm.get_available_axis(workspace)
            self._cut_view.populate_cut_axis_options(axis)
            self._cut_view.enable()

        elif self._cut_algorithm.is_cut(workspace):
            self._acting_on = "existing_cut"
            self._cut_view.plotting_params_only()
            cut_axis, integration_limits, is_normed = self._cut_algorithm.get_cut_params(workspace)
            if is_normed:
                self._cut_view.force_normalization()
            self._cut_view.populate_cut_axis_options([cut_axis.units])
            format_ = lambda *args: map(lambda x:"%.5f" % x, args)
            self._cut_view.populate_cut_params(*format_(cut_axis.start, cut_axis.end, cut_axis.step))
            self._cut_view.populate_integration_params(*format_(*integration_limits))

        else:
            self._cut_view.disable()
