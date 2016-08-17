from powder_ui import Ui_Form
from PyQt4.QtGui import QWidget
from presenters.powder_projection_presenter import PowderProjectionPresenter
from models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator
from views.powder_projection_view import PowderView
from command import Command


class PowderWidget(QWidget,Ui_Form,PowderView):
    """This widget is not usable without a main window which implements mainview"""
    def __init__(self,*args, **kwargs):
        super(PowderWidget,self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnPowderCalculateProjection.clicked.connect(self._btn_clicked)

    def set_workspace_selector(self, workspace_selector):
        # Currently will raise an error if a workspace_selector does not implement mainView
        # The code needs to be refactored and proper interface needs to be defined.
        self._presenter = PowderProjectionPresenter(self, workspace_selector, MantidProjectionCalculator())

    def _btn_clicked(self):
        self._presenter.notify(Command.CalculatePowderProjection)

    def get_powder_u1(self):
        return str(self.cmbPowderU1.currentText())

    def get_powder_u2(self):
        return str(self.cmbPowderU2.currentText())

    def populate_powder_u1(self, u1_options):
        self.cmbPowderU1.clear()
        for value in u1_options:
            self.cmbPowderU1.addItem(value)

    def populate_powder_u2(self, u2_options):
        self.cmbPowderU2.clear()
        for value in u2_options:
            self.cmbPowderU2.addItem(value)

    def populate_powder_projection_units(self, powder_projection_units):
        self.cmbPowderUnits.clear()
        for unit in powder_projection_units:
            self.cmbPowderUnits.addItem(unit)

    def get_powder_units(self):
        return str(self.cmbPowderUnits.currentText())