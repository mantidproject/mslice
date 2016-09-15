from PyQt4.QtGui import QWidget
from PyQt4.QtCore import pyqtSignal
from command import Command
from models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter
from models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
from presenters.slice_plotter_presenter import SlicePlotterPresenter
from slice_ui import Ui_Form
from views.slice_plotter_view import SlicePlotterView
import plotting.pyplot


class SliceWidget(QWidget, Ui_Form, SlicePlotterView):

    error_occurred = pyqtSignal('QString')

    def __init__(self, *args, **kwargs):
        """This Widget provides basic control over displaying slices. This widget is NOT USABLE without a main window

        The main window must implement MainView"""
        super(SliceWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnSliceDisplay.clicked.connect(self._btn_clicked)
        self.display_errors_to_statusbar = True
        plotter = MatplotlibSlicePlotter(MantidSliceAlgorithm())
        self._presenter = SlicePlotterPresenter( self, plotter )

    def get_presenter(self):
        return self._presenter

    def _btn_clicked(self):
        self._presenter.notify(Command.DisplaySlice)

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)

    def get_slice_x_axis(self):
        return str(self.cmbSliceXAxis.currentText())

    def get_slice_y_axis(self):
        return str(self.cmbSliceYAxis.currentText())

    def get_slice_is_norm_to_one(self):
        return self.rdoSliceNormToOne.isChecked()

    def get_slice_smoothing(self):
        return str(self.lneSliceSmoothing.text())

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

    def populate_slice_x_options(self, options):
        self.cmbSliceXAxis.clear()
        for option in options:
            self.cmbSliceXAxis.addItem(option)

    def populate_slice_y_options(self, options):
        self.cmbSliceYAxis.clear()
        for option in options:
            self.cmbSliceYAxis.addItem(option)

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

    def error(self, str):
        self._display_error(str)

    def populate_slice_x_params(self, x_start, x_end, x_step):
        self.lneSliceXStart.setText(x_start)
        self.lneSliceXEnd.setText(x_end)
        self.lneSliceXStep.setText(x_step)

    def populate_slice_y_params(self, y_start, y_end, y_step):
        self.lneSliceYStart.setText(y_start)
        self.lneSliceYEnd.setText(y_end)
        self.lneSliceYStep.setText(y_step)

    def clear_input_fields(self):
        self.populate_slice_x_options([])
        self.populate_slice_y_options([])
        self.populate_slice_x_params("", "", "")
        self.populate_slice_y_params("", "", "")
        self.lneSliceIntensityStart.setText("")
        self.lneSliceIntensityEnd.setText("")
        self.lneSliceSmoothing.setText("")
        self.rdoSliceNormToOne.setChecked(0)