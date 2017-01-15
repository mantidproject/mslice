from mslice.models.cut.cut_algorithm import CutAlgorithm
from mslice.models.cut.cut_plotter import CutPlotter
from mslice.presenters.slice_plotter_presenter import Axis
from mslice.views.cut_view import CutView
from mslice.widgets.cut.command import Command
from .validation_decorators import require_main_presenter
from PyQt4.QtGui import QFileDialog
from os.path import splitext
import numpy as np

class CutPresenter(object):
    def __init__(self, cut_view, cut_algorithm, cut_plotter):
        self._cut_view = cut_view
        self._cut_algorithm = cut_algorithm
        assert isinstance(cut_view, CutView)
        assert isinstance(cut_algorithm, CutAlgorithm)
        assert isinstance(cut_plotter, CutPlotter)
        self._main_presenter = None
        self._cut_plotter = cut_plotter
        self._acting_on = None
        self._cut_view.disable()

    def register_master(self, main_presenter):
        self._main_presenter = main_presenter
        self._main_presenter.subscribe_to_workspace_selection_monitor(self)

    @require_main_presenter
    def notify(self, command):
        self._clear_displayed_error()
        if command == Command.Plot:
            self._process_cuts(plot_over=False)
        elif command == Command.PlotOver:
            self._process_cuts(plot_over=True)
        elif command == Command.SaveToWorkspace:
            self._process_cuts(save_to_workspace=True)
        elif command == Command.SaveToAscii:
            fname = self.get_filename_to_save()
            self._process_cuts(save_to_file=fname)

    def _process_cuts(self, plot_over=False, save_to_workspace=False, save_to_file=None):
        """This function handles the width parameter. If it is not specified a single cut is plotted from
        integration_start to integration_end """
        try:
            params = self._parse_input()
        except ValueError:
            return

        if save_to_workspace:
            def save_cut(params, _, __):
                self._save_cut_to_workspace(params)
            handler = save_cut
        elif save_to_file is not None:
            def save_file(params, _, save_to_file):
                self._save_cut_to_file(params, save_to_file)
            handler = save_file
        else:
            def plot_cut(params, plot_over, _):
                self._plot_cut(params, plot_over)
            handler = plot_cut
        width = params[-1]
        params = params[:-1]
        if width is None:
            # No width specified, just plot a single cut
            handler(params, plot_over, save_to_file)
            return
        integration_start, integration_end = params[2:4]
        cut_start, cut_end = integration_start, min(integration_start + width, integration_end)
        index = 0
        while cut_start != cut_end:
            params = params[:2] + (cut_start, cut_end) + params[4:]
            if save_to_file is not None:
                filename, file_extension = splitext(save_to_file)
                output_file_part = filename+'_'+str(index)+file_extension
            else:
                output_file_part = None
            index += 1
            handler(params, plot_over, output_file_part)
            cut_start, cut_end = cut_end, min(cut_end + width, integration_end)
            # The first plot will respect which button the user pressed. The rest will over plot
            plot_over = True

    def _plot_cut(self, params, plot_over):
        self._cut_plotter.plot_cut(*params, plot_over=plot_over)

    def _save_cut_to_file(self, params, output_file):
        cut_params = params[:5]
        x, y, e = self._cut_algorithm.compute_cut_xye(*cut_params)
        header = 'MSlice Cut of workspace "%s" along "%s" between %f and %f' % (params[:4])
        header += ' %s normalising to unity' % ('with' if params[4] else 'without')
        self.save_data_to_txt(x, y, e, header)

    def get_filename_to_save(self):
        path = QFileDialog.getSaveFileName(caption='Select File for Saving')
        return str(path)

    def save_data_to_txt(self, x, y, e, header=None):
        out_data = np.c_[x, y, e]
        np.savetxt(str(output_file), out_data, fmt='%12.9e', header=header)

    def _save_cut_to_workspace(self, params):
        cut_params = params[:5]
        cut_ws = self._cut_algorithm.compute_cut(*cut_params)
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

        if None not in (cut_axis.start, cut_axis.end) and cut_axis.start >= cut_axis.end:
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
        return selected_workspace, cut_axis, integration_start, integration_end, norm_to_one, intensity_start, \
            intensity_end, width

    def _to_float(self, x):
        if x == "":
            return None
        return float(x)

    def workspace_selection_changed(self):
        self._cut_view.clear_input_fields()
        workspace_selection = self._main_presenter.get_selected_workspaces()

        if len(workspace_selection) != 1:
            self._cut_view.disable()
            return

        workspace = workspace_selection[0]
        if self._cut_algorithm.is_cuttable(workspace):
            axis = self._cut_algorithm.get_available_axis(workspace)
            self._cut_view.populate_cut_axis_options(axis)
            self._cut_view.enable()

        elif self._cut_algorithm.is_cut(workspace):
            self._cut_view.plotting_params_only()
            cut_axis, integration_limits, is_normed = self._cut_algorithm.get_cut_params(workspace)
            if is_normed:
                self._cut_view.force_normalization()
            self._cut_view.populate_cut_axis_options([cut_axis.units])
            def format_(*args):
                return map(lambda x: "%.5f" % x, args)
            self._cut_view.populate_cut_params(*format_(cut_axis.start, cut_axis.end, cut_axis.step))
            self._cut_view.populate_integration_params(*format_(*integration_limits))

        else:
            self._cut_view.disable()

    def _clear_displayed_error(self):
        self._cut_view.clear_displayed_error()
