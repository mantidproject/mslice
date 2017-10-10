"""A widget for cut calculations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from PyQt4.QtCore import pyqtSignal
from PyQt4.QtGui import QWidget

from mslice.models.cut.mantid_cut_algorithm import MantidCutAlgorithm
from mslice.models.cut.matplotlib_cut_plotter import MatplotlibCutPlotter
from mslice.presenters.cut_presenter import CutPresenter
from mslice.views.cut_view import CutView
from .command import Command
from .cut_ui import Ui_Form

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class CutWidget(QWidget, CutView, Ui_Form):
    error_occurred = pyqtSignal('QString')
    busy = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super(CutWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self._command_lookup = {
            self.btnCutPlot: Command.Plot,
            self.btnCutPlotOver: Command.PlotOver,
            self.btnCutSaveToWorkspace: Command.SaveToWorkspace,
            self.btnCutSaveAscii: Command.SaveToAscii
        }
        for button in self._command_lookup.keys():
            button.clicked.connect(self._btn_clicked)
        cut_algorithm = MantidCutAlgorithm()
        cut_plotter = MatplotlibCutPlotter(cut_algorithm)
        self._presenter = CutPresenter(self, cut_algorithm, cut_plotter)
        self.cmbCutAxis.currentIndexChanged.connect(self.axis_changed)
        self._minimumStep = None
        self.lneCutStep.editingFinished.connect(self._step_edited)

    def _btn_clicked(self):
        sender = self.sender()
        command = self._command_lookup[sender]
        self._presenter.notify(command)

    def _step_edited(self):
        """Checks that user inputted step size is not too small."""
        if self._minimumStep:
            try:
                value = float(self.lneCutStep.text())
            except ValueError:
                return
            if value == 0:
                self.lneCutStep.setText('%.5f' % (self._minimumStep))
                self._display_error('Setting step size to default.')
            elif value < (self._minimumStep / 10000.):
                self._display_error('Step size too small!')
                return False
        return True

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)

    def axis_changed(self, _changed_index):
        self._presenter.notify(Command.AxisChanged)

    def get_presenter(self):
        return self._presenter

    def get_cut_axis(self):
        return str(self.cmbCutAxis.currentText())

    def get_cut_axis_start(self):
        return str(self.lneCutStart.text())

    def get_cut_axis_step(self):
        return str(self.lneCutStep.text())

    def get_cut_axis_end(self):
        return str(self.lneCutEnd.text())

    def get_integration_start(self):
        return str(self.lneCutIntegrationStart.text())

    def get_integration_end(self):
        return str(self.lneCutIntegrationEnd.text())

    def get_integration_width(self):
        return str(self.lneCutIntegrationWidth.text())

    def get_intensity_start(self):
        return str(self.lneEditCutIntensityStart.text())

    def get_intensity_end(self):
        return str(self.lneCutIntensityEnd.text())

    def get_intensity_is_norm_to_one(self):
        return self.rdoCutNormToOne.isChecked()

    def get_smoothing(self):
        return str(self.lneCutSmoothing.text())

    def set_cut_axis(self, axis_name):
        index = [ind for ind in range(self.cmbCutAxis.count()) if str(self.cmbCutAxis.itemText(ind)) == axis_name]
        if index:
            self.cmbCutAxis.blockSignals(True)
            self.cmbCutAxis.setCurrentIndex(index[0])
            self.cmbCutAxis.blockSignals(False)

    def set_minimum_step(self, value):
        self._minimumStep = value

    def populate_cut_axis_options(self, options):
        self.cmbCutAxis.blockSignals(True)
        self.cmbCutAxis.clear()
        for option in options:
            self.cmbCutAxis.addItem(option)
        self.cmbCutAxis.blockSignals(False)

    def populate_cut_params(self, cut_start=None, cut_end=None, cut_step=None):
        if cut_start is not None:
            self.lneCutStart.setText(cut_start)
        if cut_end is not None:
            self.lneCutEnd.setText(cut_end)
        if cut_step is not None:
            self.lneCutStep.setText(cut_step)

    def populate_integration_params(self, integration_start=None, integration_end=None):
        if integration_start is not None:
            self.lneCutIntegrationStart.setText(integration_start)
        if integration_end is not None:
            self.lneCutIntegrationEnd.setText(integration_end)

    def clear_input_fields(self, **kwargs):
        if 'keep_axes' not in list(kwargs.keys()) or not kwargs['keep_axes']:
            self.populate_cut_axis_options([])
        self.populate_cut_params("", "", "")
        self.populate_integration_params("", "")
        self.lneCutIntegrationWidth.setText("")
        self.lneCutSmoothing.setText("")
        self.rdoCutNormToOne.setChecked(0)

    def populate_input_fields(self, saved_input):
        self.populate_cut_params(*saved_input['cut_parameters'])
        self.populate_integration_params(*saved_input['integration_range'])
        self.lneCutIntegrationWidth.setText(saved_input['integration_width'])
        self.lneCutSmoothing.setText(saved_input['smoothing'])
        self.rdoCutNormToOne.setChecked(saved_input['normtounity'])

    def get_input_fields(self):
        saved_input = dict()
        saved_input['axes'] = [str(self.cmbCutAxis.itemText(ind)) for ind in range(self.cmbCutAxis.count())]
        saved_input['cut_parameters'] = [self.get_cut_axis_start(),
                                         self.get_cut_axis_end(),
                                         self.get_cut_axis_step()]
        saved_input['integration_range'] = [self.get_integration_start(),
                                            self.get_integration_end()]
        saved_input['integration_width'] = self.get_integration_width()
        saved_input['smoothing'] = self.get_smoothing()
        saved_input['normtounity'] = self.get_intensity_is_norm_to_one()
        return saved_input

    def enable(self):
        self.lneCutStart.setEnabled(True)
        self.lneCutEnd.setEnabled(True)
        self.lneCutStep.setEnabled(True)
        self.cmbCutAxis.setEnabled(True)

        self.lneCutIntegrationStart.setEnabled(True)
        self.lneCutIntegrationEnd.setEnabled(True)
        self.lneCutIntegrationWidth.setEnabled(True)

        self.lneEditCutIntensityStart.setEnabled(True)
        self.lneCutIntensityEnd.setEnabled(True)
        self.rdoCutNormToOne.setEnabled(True)

        self.btnCutSaveToWorkspace.setEnabled(False)
        self.btnCutPlot.setEnabled(False)
        self.btnCutPlotOver.setEnabled(False)

        self.btnCutSaveAscii.setEnabled(True)
        self.btnCutSaveToWorkspace.setEnabled(True)
        self.btnCutPlot.setEnabled(True)
        self.btnCutPlotOver.setEnabled(True)

    def disable(self):
        self.lneCutStart.setEnabled(False)
        self.lneCutEnd.setEnabled(False)
        self.lneCutStep.setEnabled(False)
        self.cmbCutAxis.setEnabled(False)

        self.lneCutIntegrationStart.setEnabled(False)
        self.lneCutIntegrationEnd.setEnabled(False)
        self.lneCutIntegrationWidth.setEnabled(False)

        self.lneEditCutIntensityStart.setEnabled(False)
        self.lneCutIntensityEnd.setEnabled(False)
        self.rdoCutNormToOne.setEnabled(False)

        self.btnCutSaveAscii.setEnabled(False)
        self.btnCutSaveToWorkspace.setEnabled(False)
        self.btnCutPlot.setEnabled(False)
        self.btnCutPlotOver.setEnabled(False)

    def plotting_params_only(self):
        self.disable()
        self.lneEditCutIntensityStart.setEnabled(True)
        self.lneCutIntensityEnd.setEnabled(True)
        self.rdoCutNormToOne.setEnabled(True)

        self.btnCutPlot.setEnabled(True)
        self.btnCutPlotOver.setEnabled(True)

    def force_normalization(self):
        self.rdoCutNormToOne.setEnabled(False)
        self.rdoCutNormToOne.setChecked(True)

    def error_invalid_width(self):
        self._display_error("Invalid value for cut width")

    def error_current_selection_invalid(self):
        self._display_error("Cutting for the current workspace selection is not supported")

    def error_select_a_workspace(self):
        self._display_error("Please select a workspace to cut")

    def error_invalid_cut_axis_parameters(self):
        self._display_error("Invalid cut axis parameters")

    def error_invalid_integration_parameters(self):
        self._display_error("Invalid parameters for integration")

    def error_invalid_intensity_parameters(self):
        self._display_error("Invalid intensity range")

    def clear_displayed_error(self):
        self._display_error("")
