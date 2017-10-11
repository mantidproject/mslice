"""A widget for defining slice calculations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)
from PyQt4.QtGui import QWidget
from PyQt4.QtCore import pyqtSignal

from mslice.models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter
from mslice.models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.views.slice_plotter_view import SlicePlotterView
from .command import Command
from .slice_ui import Ui_Form


# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class SliceWidget(QWidget, Ui_Form, SlicePlotterView):
    error_occurred = pyqtSignal('QString')
    busy = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        """This Widget provides basic control over displaying slices. This widget is NOT USABLE without a main window

        The main window must implement MainView"""
        super(SliceWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnSliceDisplay.clicked.connect(self._btn_clicked)
        self.display_errors_to_statusbar = True
        plotter = MatplotlibSlicePlotter(MantidSliceAlgorithm())
        self._presenter = SlicePlotterPresenter(self, plotter)
        # Each time the fields are populated, set a minimum step size
        self._minimumStep = {}
        self.lneSliceXStep.editingFinished.connect(lambda: self._step_edited('x', self.lneSliceXStep))
        self.lneSliceYStep.editingFinished.connect(lambda: self._step_edited('y', self.lneSliceYStep))

    def get_presenter(self):
        return self._presenter

    def _btn_clicked(self):
        if self._step_edited('x', self.lneSliceXStep) and self._step_edited('y', self.lneSliceXStep):
            self._presenter.notify(Command.DisplaySlice)

    def _step_edited(self, idx, lineEdit):
        """Checks that user inputted step size is not too small."""
        if self._minimumStep:
            try:
                value = float(lineEdit.text())
            except ValueError:
                return
            if value == 0:
                lineEdit.setText(str(self._minimumStep[idx]))
                self._display_error('Setting step size to default.')
            elif value < (self._minimumStep[idx] / 10000.):
                self._display_error('Step size too small!')
                return False
        return True

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

    def populate_colormap_options(self, colormaps):
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

    def error(self, string):
        self._display_error(string)

    def populate_slice_x_params(self, x_start, x_end, x_step):
        self.lneSliceXStart.setText(x_start)
        self.lneSliceXEnd.setText(x_end)
        self.lneSliceXStep.setText(x_step)
        if x_step:
            self._minimumStep['x'] = float(x_step)

    def populate_slice_y_params(self, y_start, y_end, y_step):
        self.lneSliceYStart.setText(y_start)
        self.lneSliceYEnd.setText(y_end)
        self.lneSliceYStep.setText(y_step)
        if y_step:
            self._minimumStep['y'] = float(y_step)

    def clear_input_fields(self):
        self.populate_slice_x_options([])
        self.populate_slice_y_options([])
        self.populate_slice_x_params("", "", "")
        self.populate_slice_y_params("", "", "")
        self.lneSliceIntensityStart.setText("")
        self.lneSliceIntensityEnd.setText("")
        self.lneSliceSmoothing.setText("")
        self.rdoSliceNormToOne.setChecked(0)
        self._minimumStep = {}

    def disable(self):
        self.cmbSliceXAxis.setEnabled(False)
        self.cmbSliceYAxis.setEnabled(False)
        self.lneSliceXStart.setEnabled(False)
        self.lneSliceXEnd.setEnabled(False)
        self.lneSliceXStep.setEnabled(False)
        self.lneSliceYStart.setEnabled(False)
        self.lneSliceYEnd.setEnabled(False)
        self.lneSliceYStep.setEnabled(False)
        self.lneSliceIntensityStart.setEnabled(False)
        self.lneSliceIntensityEnd.setEnabled(False)
        self.lneSliceSmoothing.setEnabled(False)
        self.rdoSliceNormToOne.setEnabled(False)
        self.btnSliceDisplay.setEnabled(False)
        self.cmbSliceColormap.setEnabled(False)

    def enable(self):
        self.cmbSliceXAxis.setEnabled(True)
        self.cmbSliceYAxis.setEnabled(True)
        self.lneSliceXStart.setEnabled(True)
        self.lneSliceXEnd.setEnabled(True)
        self.lneSliceXStep.setEnabled(True)
        self.lneSliceYStart.setEnabled(True)
        self.lneSliceYEnd.setEnabled(True)
        self.lneSliceYStep.setEnabled(True)
        self.lneSliceIntensityStart.setEnabled(True)
        self.lneSliceIntensityEnd.setEnabled(True)
        self.lneSliceSmoothing.setEnabled(True)
        self.rdoSliceNormToOne.setEnabled(True)
        self.btnSliceDisplay.setEnabled(True)
        self.cmbSliceColormap.setEnabled(True)

    def clear_displayed_error(self):
        self._display_error("")
