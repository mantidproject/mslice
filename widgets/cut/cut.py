from PyQt4.QtGui import QWidget
from PyQt4.QtCore import pyqtSignal
from cut_ui import Ui_Form
from views.cut_view import CutView
from command import Command
from presenters.cut_presenter import CutPresenter
from models.cut.matplotlib_cut_plotter import MatplotlibCutPlotter
from models.cut.mantid_cut_algorithm import MantidCutAlgorithm


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
        plotter = MatplotlibCutPlotter(MantidCutAlgorithm())
        self._presenter = CutPresenter(self, plotter)

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
        return str(self.lneCutIntegrationEnd.text())

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

    def error_select_a_workspace(self):
        self._display_error("Please select a workspace to cut")