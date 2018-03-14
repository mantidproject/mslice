from __future__ import (absolute_import, division, print_function)

from mslice.models.cut.cut_algorithm import CutAlgorithm
from mslice.models.cut.cut_plotter import CutPlotter
from mslice.presenters.presenter_utility import PresenterUtility
from mslice.presenters.slice_plotter_presenter import Axis
from mslice.views.cut_view import CutView
from mslice.widgets.cut.command import Command
from .validation_decorators import require_main_presenter
import warnings


class CutPresenter(PresenterUtility):
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
        self._previous_cut = None
        self._previous_axis = None
        self._minimumStep = dict()

    @require_main_presenter
    def notify(self, command):
        self._clear_displayed_error(self._cut_view)
        self._cut_view.busy.emit(True)
        if command == Command.Plot:
            self._cut(output_method=self._plot_and_save_to_workspace)
        elif command == Command.PlotOver:
            self._cut(output_method=self._plot_and_save_to_workspace, plot_over=True)
        elif command == Command.PlotFromWorkspace:
            self._plot_cut_from_workspace(plot_over=False)
        elif command == Command.PlotOverFromWorkspace:
            self._plot_cut_from_workspace(plot_over=True)
        elif command == Command.SaveToWorkspace:
            self._cut(output_method=self._save_cut_to_workspace)
        elif command == Command.AxisChanged:
            self._cut_axis_changed()
        elif command == Command.IntegrationAxisChanged:
            self._integration_axis_changed()
        self._cut_view.busy.emit(False)

    def _cut(self, output_method, plot_over=False):
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        try:
            self._parse_step()
            parsed_params = self._parse_input()
        except ValueError:
            return
        for workspace in selected_workspaces:
            params = (workspace,) + parsed_params
            self._run_cut_method(params, output_method, plot_over)
            plot_over = True # The first plot will respect which button the user pressed. The rest will over plot

    def _run_cut_method(self, params, output_method, plot_over=False):
            width = params[-1]
            params = params[:-1]
            if width is not None:
                self._plot_with_width(params, output_method, width, plot_over)
            else:
                output_method(params, plot_over)

    def _plot_with_width(self, params, output_method, width, plot_over):
        """This function handles the width parameter."""
        integration_start = params[2].start
        integration_end = params[2].end
        cut_start, cut_end = integration_start, min(integration_start + width, integration_end)
        index = 0
        while cut_start != cut_end:
            params = params[:2] + (Axis(params[2].units, cut_start, cut_end, 0.),) + params[3:]
            output_method(params, plot_over)
            index += 1
            cut_start, cut_end = cut_end, min(cut_end + width, integration_end)
            # The first plot will respect which button the user pressed. The rest will over plot
            plot_over = True

    def _plot_and_save_to_workspace(self, params, plot_over):
        self._plot_cut(params, plot_over)
        self._save_cut_to_workspace(params, plot_over)

    def _plot_cut(self, params, plot_over):
        self._cut_plotter.plot_cut(*params, plot_over=plot_over)
        self._main_presenter.highlight_ws_tab(2)

    def _save_cut_to_workspace(self, params, _):
        cut_params = params[:4]
        self._cut_plotter.save_cut(cut_params)
        self._main_presenter.update_displayed_workspaces()

    def _plot_cut_from_workspace(self, plot_over):
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        for workspace in selected_workspaces:
            x, y, e, units = self._cut_algorithm.get_arrays_from_workspace(workspace)
            self._cut_plotter.plot_cut_from_xye(x, y, e, units, workspace, plot_over=plot_over)
            plot_over = True  # plot over if multiple workspaces selected

    def _parse_step(self):
        step = self._cut_view.get_cut_axis_step()
        try:
            step = float(step)
        except ValueError:
            step = self._cut_view.get_minimum_step()
            if step is not None:
                self._cut_view.populate_cut_params(self._cut_view.get_cut_axis_start(),
                                                   self._cut_view.get_cut_axis_end(),
                                                   "%0.5f" % step)
                self._cut_view.error_invalid_cut_step_parameter()
                warnings.warn("Invalid cut step, using default value")

    def _parse_input(self):
        # The messages of the raised exceptions are discarded. They are there for the sake of clarity/debugging
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

        integration_axis = Axis(self._cut_view.get_integration_axis(), self._cut_view.get_integration_start(),
                                self._cut_view.get_integration_end(), 0.)
        try:
            integration_axis.start = float(integration_axis.start)
            integration_axis.end = float(integration_axis.end)
        except ValueError:
            self._cut_view.error_invalid_integration_parameters()
            raise ValueError("Invalid integration parameters")

        if None not in (integration_axis.start, integration_axis.end) and integration_axis.start >= integration_axis.end:
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
        return cut_axis, integration_axis, norm_to_one, intensity_start, intensity_end, width

    def _set_minimum_step(self, workspace, axis):
        """Gets axes limits from workspace_provider and then sets the minimumStep dictionary with those values"""
        for ax in axis:
            try:
                self._minimumStep[ax] = self._cut_algorithm.get_axis_range(workspace, ax)[2]
            except (KeyError, RuntimeError):
                self._minimumStep[ax] = None
        self._cut_view.set_minimum_step(self._minimumStep[axis[0]])

    def workspace_selection_changed(self):
        if self._previous_cut is not None and self._previous_axis is not None:
            if not self._cut_view.is_fields_cleared():
                self._cut_algorithm.set_saved_cut_parameters(self._previous_cut, self._previous_axis,
                                                             self._cut_view.get_input_fields())
            else:
                self._previous_cut = None
                self._previous_axis = None
        workspace_selection = self._main_presenter.get_selected_workspaces()
        if len(workspace_selection) < 1:
            self._cut_view.clear_input_fields()
            self._cut_view.disable()
            self._previous_cut = None
            self._previous_axis = None
            return
        self._populate_fields_using_workspace(workspace_selection[0])

    def _populate_fields_using_workspace(self, workspace, plotting=False):
        if self._cut_algorithm.is_cuttable(workspace):
            axis = self._cut_algorithm.get_available_axis(workspace)
            # There are three choices for which axes to select:
            #   1. If the current cut is of the same type (e.g. QE), and parameters for the current
            #      axis in the new cut has not been defined by the user, use the current axis
            #   2. If the user has looked at this cut _and_ this axis before, use that
            #   3. Otherwise use the first available axis
            this_cut_par, prev_selected_axis = self._cut_algorithm.get_saved_cut_parameters(workspace)
            prev_cut_par, _ = self._cut_algorithm.get_saved_cut_parameters(self._previous_cut, self._previous_axis)
            axis_is_same_as_prev = prev_cut_par is not None and axis == prev_cut_par['axes']
            axis_in_dict = self._cut_algorithm.is_axis_saved(workspace, self._previous_axis)
            if axis_is_same_as_prev and not axis_in_dict:
                current_axis = self._cut_view.get_cut_axis()
                saved_parameters = prev_cut_par
            elif this_cut_par is not None:
                current_axis = prev_selected_axis
                saved_parameters = this_cut_par
            else:
                current_axis = axis[0]
                saved_parameters = None

            self._cut_view.clear_input_fields()
            self._cut_view.populate_cut_axis_options(axis)
            self._cut_view.enable()
            self._cut_view.set_cut_axis(current_axis)
            if not plotting and saved_parameters is not None:
                self._cut_view.populate_input_fields(saved_parameters)
            self._previous_cut = workspace
            self._previous_axis = current_axis
            self._set_minimum_step(workspace, axis)
        else:
            self._cut_view.clear_input_fields()
            self._cut_view.disable()
            self._previous_cut = None
            self._previous_axis = None

    def _cut_axis_changed(self):
        if self._previous_axis is not None and not self._cut_view.is_fields_cleared():
            self._cut_algorithm.set_saved_cut_parameters(self._previous_cut, self._previous_axis,
                                                         self._cut_view.get_input_fields())
        self._cut_view.clear_input_fields(keep_axes=True)
        if self._previous_cut is not None:
            self._previous_axis = self._cut_view.get_cut_axis()
            saved_parameters, _ = self._cut_algorithm.get_saved_cut_parameters(self._previous_cut, self._previous_axis)
            if saved_parameters is not None:
                self._cut_view.populate_input_fields(saved_parameters)
        min_step = self._minimumStep[self._cut_view.get_cut_axis()]
        self._cut_view.set_minimum_step(min_step)
        self._cut_view.update_integration_axis()

    def set_workspace_provider(self, workspace_provider):
        self._cut_plotter.set_workspace_provider(workspace_provider)
