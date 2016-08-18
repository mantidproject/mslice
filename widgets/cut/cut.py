from cut_ui import Ui_Form
from views.cut_view import CutView
from PyQt4.QtGui import QWidget


class CutWidget(QWidget, CutView, Ui_Form):
    def __init__(self,*args, **kwargs):
        super(CutWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)


    def set_workspace_selector(self, workspace_selector):
        pass

    def get_cut_axis(self):
        return str(self.cmbCutAxis.currentText())

    def get_cut_axis_start(self):
        return str(self.lneCutStart.text())

    def get_cut_axis_step(self):
        return str(self.lneCutStep.text())

    def get_cut_axis_end(self):
        return str(self.lneCutEnd.text())

    def get_integration_axis_start(self):
        return str(self.lneCutIntegrationStart.text())

    def get_integration_axis_end(self):
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