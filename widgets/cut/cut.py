from PyQt4.QtGui import QWidget
from PyQt4.QtCore import pyqtSignal
from cut_ui import Ui_Form
from views.cut_view import CutView
from command import Command
from presenters.cut_presenter import CutPresenter
from models.cut.mantid_cut_algorithm import MantidCutAlgorithm
from models.cut.matplotlib_cut_plotter import MatplotlibCutPlotter
import plotting.pyplot


class CutWidget(QWidget, CutView, Ui_Form):
    error_occurred = pyqtSignal('QString')

    def __init__(self,*args, **kwargs):
        super(CutWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
        self._command_lookup = {
            self.btnCutPlot: Command.Plot,
            self.btnCutPlotOver: Command.PlotOver,
            self.btnCutSaveToWorkspace: Command.SaveToWorkspace,
            self.btnCutSaveAscii: Command.SaveToAscii
        }
        for button in self._command_lookup.keys():
            button.clicked.connect(self._btn_clicked)
        cut_alogrithm = MantidCutAlgorithm()
        cut_plotter = MatplotlibCutPlotter(cut_alogrithm)
        self._presenter = CutPresenter(self, cut_alogrithm, cut_plotter)

    def _btn_clicked(self):
        sender = self.sender()
        command = self._command_lookup[sender]
        self._presenter.notify(command)

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)

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

    def populate_cut_axis_options(self,options):
        self.cmbCutAxis.clear()
        for option in options:
            self.cmbCutAxis.addItem(option)

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

    def clear_input_fields(self):
        self.populate_cut_axis_options([])
        self.populate_cut_params("", "", "")
        self.populate_integration_params("", "")
        self.lneCutIntegrationWidth.setText("")
        self.lneCutSmoothing.setText("")
        self.rdoCutNormToOne.setChecked(0)


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