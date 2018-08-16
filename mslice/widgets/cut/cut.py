"""A widget for cut calculations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)

from mslice.util.qt import load_ui
from mslice.util.qt.QtGui import QDoubleValidator
from mslice.util.qt.QtCore import Signal
from mslice.util.qt.QtWidgets import QWidget

from mslice.presenters.cut_widget_presenter import CutWidgetPresenter

from mslice.views.interfaces.cut_view import CutView
from .command import Command


# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class CutWidget(CutView, QWidget):
    error_occurred = Signal('QString')
    busy = Signal(bool)

    def __init__(self, parent=None, *args, **kwargs):
        QWidget.__init__(self, parent, *args, **kwargs)
        load_ui(__file__, 'cut.ui', self)
        self._command_lookup = {
            self.btnCutPlot: Command.Plot,
            self.btnCutPlotOver: Command.PlotOver,
            self.btnCutSaveToWorkspace: Command.SaveToWorkspace
        }
        for button in self._command_lookup.keys():
            button.clicked.connect(self._btn_clicked)
        self._presenter = CutWidgetPresenter(self)
        self.cmbCutAxis.currentIndexChanged.connect(self.axis_changed)
        self._minimumStep = None
        self.lneCutStep.editingFinished.connect(self._step_edited)
        self.enable_integration_axis(False)
        self.set_validators()

    def _btn_clicked(self):
        sender = self.sender()
        command = self._command_lookup[sender]
        if self._step_edited():
            self._presenter.notify(command)

    def _step_edited(self):
        """Checks that user inputted step size is not too small."""
        if self._minimumStep:
            try:
                value = float(self.lneCutStep.text())
            except ValueError:
                value = 0.0
                self.display_error('Invalid cut step parameter. Using default.')
            if value == 0.0:
                self.lneCutStep.setText('%.5f' % (self._minimumStep))
                self.display_error('Setting step size to default.')
            elif value < (self._minimumStep / 100.):
                self.display_error('Step size too small!')
                return False
        return True

    def display_error(self, error_string):
        self.error_occurred.emit(error_string)

    def axis_changed(self, _changed_index):
        self._presenter.notify(Command.AxisChanged)

    def enable_integration_axis(self, enabled):
        if enabled:
            self.integrationStack.setCurrentIndex(1)
            self.label_250.show()
        else:
            self.integrationStack.setCurrentIndex(0)
            self.label_250.hide()

    def integration_axis_shown(self):
        return self.integration_axis.current_index == 1

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

    def get_integration_axis(self):
        if self.integration_axis_shown:
            return str(self.cmbIntegrationAxis.currentText())
        else:
            return None

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

    def get_minimum_step(self):
        return self._minimumStep

    def populate_cut_axis_options(self, options):
        self.cmbCutAxis.blockSignals(True)
        self.cmbCutAxis.clear()
        for option in options:
            self.cmbCutAxis.addItem(option)
        self.cmbCutAxis.blockSignals(False)

    def populate_integration_axis_options(self, options):
        self.cmbIntegrationAxis.blockSignals(True)
        self.cmbIntegrationAxis.clear()
        for option in options:
            self.cmbIntegrationAxis.addItem(option)
        self.cmbIntegrationAxis.setEnabled(len(options) > 1)
        self.cmbIntegrationAxis.blockSignals(False)

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
        if 'keep_axes' not in kwargs or not kwargs['keep_axes']:
            self.populate_cut_axis_options([])
        self.populate_cut_params("", "", "")
        self.populate_integration_params("", "")
        self.lneCutIntegrationWidth.setText("")
        self.lneCutSmoothing.setText("")
        self.rdoCutNormToOne.setChecked(0)

    def is_fields_cleared(self):
        current_fields = self.get_input_fields()
        cleared_fields = {'cut_parameters': ['', '', ''],
                          'integration_range': ['', ''],
                          'integration_width': '',
                          'smoothing': '',
                          'normtounity': False}
        for k in cleared_fields:
            if current_fields[k] != cleared_fields[k]:
                return False
        return True

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

    def set_validators(self):
        line_edits = [self.lneCutStart, self.lneCutEnd, self.lneCutIntegrationStart,
                      self.lneCutIntegrationEnd, self.lneCutIntegrationWidth, self.lneEditCutIntensityStart,
                      self.lneCutIntensityEnd]
        for line_edit in line_edits:
            line_edit.setValidator(QDoubleValidator())

    def force_normalization(self):
        self.rdoCutNormToOne.setEnabled(False)
        self.rdoCutNormToOne.setChecked(True)

    def clear_displayed_error(self):
        self.display_error("")
