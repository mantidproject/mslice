from PyQt4.QtGui import QWidget
from PyQt4.QtCore import pyqtSignal
from command import Command
from models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter
from models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
from presenters.slice_plotter_presenter import SlicePlotterPresenter
from slice_ui import Ui_Form
from views.slice_plotter_view import SlicePlotterView


class SliceWidget(QWidget, Ui_Form, SlicePlotterView):

    error_occurred = pyqtSignal('QString')

    def __init__(self, *args, **kwargs):
        """This Widget provides basic control over displaying slices. This widget is NOT USABLE without a main window

        The main window must implement MainView"""
        super(SliceWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnSliceDisplay.clicked.connect(self._btn_clicked)
        self.display_errors_to_statusbar = True


    def _btn_clicked(self):
        self._presenter.notify(Command.DisplaySlice)

    def set_workspace_selector(self, workspace_selector):
        # Currently will raise an error if a workspace_selector does not implement mainView
        # The code needs to be refactored and proper interface needs to be defined.
        slice_plotter = MatplotlibSlicePlotter(MantidSliceAlgorithm())
        self._presenter = SlicePlotterPresenter(main_view=workspace_selector, slice_view=self,
                                                slice_plotter=slice_plotter)
        #
        self._main_window = workspace_selector

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)

    def get_slice_x_start(self):
        return str(self.lneSliceXStart.text())

    def get_slice_x_end(self):
        return str(self.lneSliceXEnd.text())

    def get_slice_x_step(self):
        return str(self.lneSliceXStep.text())

    def get_slice_y_start(self):
        return str(self.lneSliceYStart.text())

    def get_slice_y_end(self):
        return str(self.lneSliceYEnd.text())

    def get_slice_y_step(self):
        return str(self.lneSliceYStep.text())

    def get_slice_colourmap(self):
        return str(self.cmbSliceColormap.currentText())

    def get_slice_intensity_start(self):
        return str(self.lneSliceIntensityStart.text())

    def get_slice_intensity_end(self):
        return str(self.lneSliceIntensityEnd.text())

    def populate_colormap_options(self,colormaps):
        self.cmbSliceColormap.clear()
        for colormap in colormaps:
            self.cmbSliceColormap.addItem(colormap)


    def error_select_one_workspace(self):
        self._display_error('Please select a workspace to slice')

    def error_invalid_x_params(self):
        self._display_error('Invalid parameters for the x axis of the slice')

    def error_invalid_intensity_params(self):
        self._display_error('Invalid parameters for the intensity of the slice')

    def error_invalid_plot_parameters(self):
        self._display_error('Invalid parameters for the slice')

    def error_invalid_smoothing_params(self):
        self._display_error('Invalid value for smoothing')

    def error_invalid_y_units(self):
        self._display_error('Invalid selection of the y axis')

    def error_invalid_y_params(self):
        self._display_error('Invalid parameters for the y axis os the slice')

    def error_invalid_x_units(self):
        self._display_error('Invalid selection of the x axis')


