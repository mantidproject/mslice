from __future__ import (absolute_import, division, print_function)
from .busy import show_busy
from mslice.models.alg_workspace_ops import get_available_axes, get_axis_range
from mslice.models.axis import Axis
from mslice.models.units import EnergyUnits
from mslice.models.cmap import ALLOWED_CMAPS
from mslice.models.slice.slice_functions import is_sliceable
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.presenters.presenter_utility import PresenterUtility
from mslice.views.interfaces.slice_view import SliceView
from mslice.widgets.slice.command import Command
from .interfaces.slice_plotter_presenter import SlicePlotterPresenterInterface
from .validation_decorators import require_main_presenter


def validate(field_method, error_method):
    try:
        return field_method()
    except ValueError as e:
        error_method()
        raise e


class SliceWidgetPresenter(PresenterUtility, SlicePlotterPresenterInterface):
    def __init__(self, slice_view):
        if not isinstance(slice_view, SliceView):
            raise TypeError("Parameter slice_view is not of type SlicePlotterView")
        self._slice_view = slice_view
        self._main_presenter = None
        self._slice_plotter_presenter = None
        self._en_default = 'meV'

    def set_slice_plotter_presenter(self, slice_plotter_presenter):
        self._slice_plotter_presenter = slice_plotter_presenter
        self._slice_view.populate_colormap_options(ALLOWED_CMAPS)

    def notify(self, command):
        self._clear_displayed_error(self._slice_view)
        with show_busy(self._slice_view):
            if command == Command.DisplaySlice:
                self._display_slice()

            else:
                raise ValueError("Slice Plotter Presenter received an unrecognised command")

    def _display_slice(self):
        try:
            selected_workspace = validate(self._selected_workspace, self._slice_view.error_select_one_workspace)
            x_axis = validate(self._x_axis, self._slice_view.error_invalid_x_params)
            y_axis = validate(self._y_axis, self._slice_view.error_invalid_y_params)
            intensity_start, intensity_end = validate(self._intensity, self._slice_view.error_invalid_intensity_params)
        except ValueError:
            return

        if x_axis.units == y_axis.units:
            self._slice_view.error_invalid_plot_parameters()
            return

        norm_to_one = bool(self._slice_view.get_slice_is_norm_to_one())
        colourmap = self._slice_view.get_slice_colourmap()

        self._plot_slice(selected_workspace, x_axis, y_axis, intensity_start, intensity_end,
                         norm_to_one, colourmap)

    def _plot_slice(self, *args):
        try:
            self._slice_plotter_presenter.plot_slice(*args)
        except RuntimeError as e:
            import traceback
            traceback.print_exc(e)
            self._slice_view.error(e.args[0])
        except ValueError as e:
            # This gets thrown by matplotlib if the supplied intensity_min > data_max_value or vise versa
            # will break if matplotlib change exception error message
            if str(e) != "minvalue must be less than or equal to maxvalue":
                raise e
            self._slice_view.error_invalid_intensity_params()

    def _selected_workspace(self):
        selected_workspaces = self._get_main_presenter().get_selected_workspaces()
        if not selected_workspaces or len(selected_workspaces) > 1:
            raise ValueError()
        return selected_workspaces[0]

    def _x_axis(self):
        return Axis(self._slice_view.get_slice_x_axis(), self._slice_view.get_slice_x_start(),
                    self._slice_view.get_slice_x_end(), self._slice_view.get_slice_x_step(),
                    self._slice_view.get_units())

    def _y_axis(self):
        return Axis(self._slice_view.get_slice_y_axis(), self._slice_view.get_slice_y_start(),
                    self._slice_view.get_slice_y_end(), self._slice_view.get_slice_y_step(),
                    self._slice_view.get_units())

    def _intensity(self):
        intensity_start = self._slice_view.get_slice_intensity_start()
        intensity_end = self._slice_view.get_slice_intensity_end()
        return self._slice_plotter_presenter.validate_intensity(intensity_start, intensity_end)

    def _smoothing(self):
        smoothing = self._slice_view.get_slice_smoothing()
        return self._to_int(smoothing)

    @require_main_presenter
    def _get_main_presenter(self):
        return self._main_presenter

    def workspace_selection_changed(self):
        workspace_selection = self._get_main_presenter().get_selected_workspaces()
        if len(workspace_selection) != 1 or not is_sliceable(workspace_selection[0]):
            self._slice_view.clear_input_fields()
            self._slice_view.disable()
        else:
            workspace_selection = workspace_selection[0]
            self._slice_view.enable()
            axis = get_available_axes(get_workspace_handle(workspace_selection))
            self._slice_view.populate_slice_x_options(axis)
            self._slice_view.populate_slice_y_options(axis[::-1])
            self.populate_slice_params()

    def populate_slice_params(self):
        try:
            workspace_selection = get_workspace_handle(self._get_main_presenter().get_selected_workspaces()[0])
            x_min, x_max, x_step = get_axis_range(workspace_selection, self._slice_view.get_slice_x_axis())
            y_min, y_max, y_step = get_axis_range(workspace_selection, self._slice_view.get_slice_y_axis())
        except (KeyError, RuntimeError, IndexError):
            self._slice_view.clear_input_fields()
            self._slice_view.disable()
        else:
            e_units = EnergyUnits(self._slice_view.get_units())
            if e_units.factor_from_meV() != 1.:
                if 'DeltaE' in self._slice_view.get_slice_x_axis():
                    x_min, x_max, x_step = (float(v) for v in e_units.from_meV(x_min, x_max, x_step))
                else:
                    y_min, y_max, y_step = (float(v) for v in e_units.from_meV(y_min, y_max, y_step))
            self._slice_view.enable()
            self._slice_view.populate_slice_x_params(*["%.5f" % x for x in (x_min, x_max, x_step)])
            self._slice_view.populate_slice_y_params(*["%.5f" % x for x in (y_min, y_max, y_step)])

    def update_workspaces(self):
        self._main_presenter.update_displayed_workspaces()

    def set_energy_default(self, en_default):
        self._en_default = en_default
        self._slice_view.set_energy_default(en_default)
        self._slice_view.set_units(en_default)
