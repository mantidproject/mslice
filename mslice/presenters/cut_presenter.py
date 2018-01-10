from __future__ import (absolute_import, division, print_function)
from mslice.models.cut.cut_algorithm import CutAlgorithm
from mslice.models.cut.cut_plotter import CutPlotter
from mslice.presenters.presenter_utility import PresenterUtility
from mslice.presenters.slice_plotter_presenter import Axis
from mslice.views.cut_view import CutView
from mslice.widgets.cut.command import Command
from .validation_decorators import require_main_presenter
from qtpy.QtWidgets import QFileDialog
from os.path import splitext
import numpy as np


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
            self._process_cuts(plot_over=False)
        elif command == Command.PlotOver:
            self._process_cuts(plot_over=True)
        elif command == Command.SaveToWorkspace:
            self._process_cuts(save_to_workspace=True)
        elif command == Command.SaveToAscii:
            fname = str(QFileDialog.getSaveFileName(caption='Select File for Saving'))
            if fname:
                self._process_cuts(save_to_file=fname)
        elif command == Command.AxisChanged:
            self._cut_axis_changed()
        self._cut_view.busy.emit(False)


    def _process_cuts(self, plot_over=False, save_to_workspace=False, save_to_file=None, workspace_index=0):
        """This function handles the width parameter. If it is not specified a single cut is plotted from
        integration_start to integration_end """
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        try:
            params = self._parse_input(workspace_index)
        except ValueError:
            return
        if save_to_workspace:
            def handler(*args):
                self._save_cut_to_workspace(args[0])
        elif save_to_file is not None:
            def handler(params, _, save_to_file):
                self._save_cut_to_file(params, save_to_file)
        else:
            def handler(params, plot_over, _):
                self._plot_cut(params, plot_over)

        width = params[-1]
        params = params[:-1]
        if width is not None:
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
        else:
            # No width specified, just plot a single cut
            handler(params, plot_over, save_to_file)

        if workspace_index < len(selected_workspaces) - 1:
            workspace_index += 1
            self._populate_fields_using_workspace(selected_workspaces[workspace_index], plotting=True)
            self._process_cuts(plot_over=True, save_to_workspace=save_to_workspace, save_to_file=save_to_file,
                               workspace_index=workspace_index)

    def _plot_cut(self, params, plot_over):
        self._cut_plotter.plot_cut(*params, plot_over=plot_over)

    def _save_cut_to_file(self, params, output_file):
        cut_params = params[:5]
        x, y, e = self._cut_algorithm.compute_cut_xye(*cut_params)
        header = 'MSlice Cut of workspace "%s" along "%s" between %f and %f' % (params[:4])
        header += ' %s normalising to unity' % ('with' if params[4] else 'without')
        out_data = np.c_[x, y, e]
        np.savetxt(str(output_file), out_data, fmt='%12.9e', header=header)

    def _save_cut_to_workspace(self, params):
        cut_params = params[:5]
        self._cut_algorithm.compute_cut(*cut_params)
        self._main_presenter.update_displayed_workspaces()

    def _parse_input(self, workspace_index=0):
        # The messages of the raised exceptions are discarded. They are there for the sake of clarity/debugging
        selected_workspaces = self._main_presenter.get_selected_workspaces()
        selected_workspace = selected_workspaces[workspace_index]
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

        elif self._cut_algorithm.is_cut(workspace):
            self._cut_view.clear_input_fields()
            self._cut_view.plotting_params_only()
            cut_axis, integration_limits, is_normed = self._cut_algorithm.get_cut_params(workspace)
            if is_normed:
                self._cut_view.force_normalization()
            self._cut_view.populate_cut_axis_options([cut_axis.units])

            def format_(*args):
                return ["%.5f" % x for x in args]
            self._cut_view.populate_cut_params(*format_(cut_axis.start, cut_axis.end, cut_axis.step))
            self._cut_view.populate_integration_params(*format_(*integration_limits))
            self._minimumStep[cut_axis.units] = None
            self._previous_cut = None
            self._previous_axis = None

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

    def set_workspace_provider(self, workspace_provider):
        self._cut_algorithm.set_workspace_provider(workspace_provider)
