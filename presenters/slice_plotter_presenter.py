from mainview import MainView
from models.slice.slice_plotter import SlicePlotter
from views.slice_plotter_view import SlicePlotterView
from widgets.slice.command import Command


class Axis:
    def __init__(self, units, start, end, step):
        self.units = units
        self.start = start
        self.end = end
        self.step = step

    def __eq__(self, other):
        # This is required for Unit testing
        return self.units == other.units and self.start == other.start and self.end == other.end \
                and self.step == other.step and isinstance(other, Axis)

    def __repr__(self):
        info = (self.units, self.start, self.end, self.step)
        return "Axis(" + " ,".join(map(str,info)) + ")"


INVALID_PARAMS = 1
INVALID_X_PARAMS = 2
INVALID_Y_PARAMS = 3
INVALID_INTENSITY = 4
INVALID_SMOOTHING = 5
INVALID_X_UNITS = 6
INVALID_Y_UNITS = 7

class SlicePlotterPresenter:
    def __init__(self, main_view, slice_view,slice_plotter):
        if not isinstance(main_view, MainView):
            raise TypeError("Parameter main_view is not of type MainView")
        if not isinstance(slice_view, SlicePlotterView):
            raise TypeError("Parameter slice_view is not of type SlicePlotterView")
        if not isinstance(slice_plotter, SlicePlotter):
            raise TypeError("Parameter slice_plotter is not of type SlicePlotter")
        self._slice_view = slice_view
        self._main_view = main_view
        self._slice_plotter = slice_plotter
        colormaps = self._slice_plotter.get_available_colormaps()
        self._slice_view.populate_colormap_options(colormaps)

    def notify(self,command):
        if command == Command.DisplaySlice:
            self._display_slice()
        else:
            raise ValueError("Slice Plotter Presenter received an unrecognised command")

    def _display_slice(self):
        selected_workspaces = self._get_main_presenter().get_selected_workspaces()
        if not selected_workspaces:
            self._slice_view.error_select_one_workspace()
            return
        if len(selected_workspaces) > 1:
            raise NotImplementedError('')

        selected_workspace = selected_workspaces[0]
        x_axis = Axis(self._slice_view.get_slice_x_axis(), self._slice_view.get_slice_x_start(),
                      self._slice_view.get_slice_x_end(), self._slice_view.get_slice_x_step())
        y_axis = Axis(self._slice_view.get_slice_y_axis(), self._slice_view.get_slice_y_start(),
                      self._slice_view.get_slice_y_end(), self._slice_view.get_slice_y_step())
        status = self._process_axis(x_axis, y_axis)
        if status == INVALID_Y_PARAMS:
            self._slice_view.error_invalid_y_params()
            return
        elif status ==INVALID_X_PARAMS:
            self._slice_view.error_invalid_x_params()
            return
        elif status ==INVALID_PARAMS:
            self._slice_view.error_invalid_plot_parameters()
            return

        intensity_start = self._slice_view.get_slice_intensity_start()
        intensity_end = self._slice_view.get_slice_intensity_end()
        norm_to_one = self._slice_view.get_slice_is_norm_to_one()
        smoothing = self._slice_view.get_slice_smoothing()
        colourmap = self._slice_view.get_slice_colourmap()
        try:
            intensity_start = self._to_float(intensity_start)
            intensity_end = self._to_float(intensity_end)
        except ValueError:
            self._slice_view.error_invalid_intensity_params()
            return

        if intensity_start > intensity_end:
            self._slice_view.error_invalid_intensity_params()
            return
        try:
            smoothing = self._to_int(smoothing)
        except ValueError:
            self._slice_view.error_invalid_smoothing_params()
        try:
            status = self._slice_plotter.display_slice(selected_workspace,x_axis, y_axis, smoothing, intensity_start,
                                                   intensity_end, norm_to_one, colourmap)
        except NotImplementedError:
            self._slice_view.error("workspace 2d slicing not supported")

        if status == INVALID_PARAMS:
            self._slice_view.error_invalid_plot_parameters()

        elif status == INVALID_X_PARAMS:
            self._slice_view.error_invalid_x_params()

        elif status == INVALID_Y_PARAMS:
            self._slice_view.error_invalid_y_params()

        elif status == INVALID_INTENSITY:
            self._slice_view.error_invalid_intensity_params()

        elif status == INVALID_SMOOTHING:
            self._slice_view.error_invalid_smoothing_params()

        elif status == INVALID_X_UNITS:
            self._slice_view.error_invalid_x_units()

        elif status == INVALID_Y_UNITS:
            self._slice_view.error_invalid_y_units()

    def _get_main_presenter(self):
        # Get the presenter when needed as opposed to initializing it as a class variable in the constructor
        # givs the flexibilty to instantiate this presenter before the main presenter
        return self._main_view.get_presenter()

    def _process_axis(self,x, y):
        if x.units == y.units:
            #return INVALID_PARAMS
            # TODO Waiting for unit lists to be populated
            pass
        try:
            x.start = self._to_float(x.start)
            x.step = self._to_float(x.step)
            x.end = self._to_float(x.end)
        except ValueError:
            return INVALID_X_PARAMS

        try:
            y.start = self._to_float(y.start)
            y.step = self._to_float(y.step)
            y.end = self._to_float(y.end)
        except ValueError:
            return INVALID_Y_PARAMS

        if x.start and x.end:
            if x.start > x.end:
                return INVALID_X_PARAMS

        if y.start and y.end:
            if y.start > y.end:
                return INVALID_Y_PARAMS

    def _to_float(self, x):
        if x is None:
            return x
        return float(x)

    def _to_int(self, x):
        if x is None:
            return x
        return int(x)