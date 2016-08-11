from powder_ui import Ui_Form
from PyQt4.QtGui import QWidget
from presenters.powder_projection_presenter import PowderProjectionPresenter
from views.powder_projection_view import PowderView
from models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator
from command import Command



class PowderWidget(QWidget,Ui_Form,PowderView):
    def __init__(self,*args, **kwargs):
        super(PowderWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnPowderCalculateProjection.clicked.connect(self._btn_clicked)

    def set_main_window(self,main_window):
        self._presenter = PowderProjectionPresenter(self,main_window,MantidProjectionCalculator())


    def _btn_clicked(self):
        self._presenter.notify(Command.CalculatePowderProjection)

    def get_powder_u1(self):
        return str(self.cmbPowderU1.currentText())

    def get_powder_u2(self):
        return str(self.cmbPowderU2.currentText())

