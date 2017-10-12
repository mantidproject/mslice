"""A widget for defining projections for powders
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from PyQt4.QtGui import QWidget
from PyQt4.QtCore import pyqtSignal

from mslice.models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator
from mslice.presenters.powder_projection_presenter import PowderProjectionPresenter
from mslice.views.powder_projection_view import PowderView
from .command import Command
from .powder_ui import Ui_Form

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class PowderWidget(QWidget, Ui_Form, PowderView):
    """This widget is not usable without a main window which implements mainview"""

    error_occurred = pyqtSignal('QString')
    busy = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super(PowderWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btnPowderCalculateProjection.clicked.connect(self._btn_clicked)
        self._presenter = PowderProjectionPresenter(self, MantidProjectionCalculator())
        self.cmbPowderU1.currentIndexChanged.connect(self._u1_changed)
        self.cmbPowderU2.currentIndexChanged.connect(self._u2_changed)

    def get_presenter(self):
        return self._presenter

    def _u1_changed(self):
        self._presenter.notify(Command.U1Changed)

    def _u2_changed(self):
        self._presenter.notify(Command.U2Changed)

    def _btn_clicked(self):
        self._presenter.notify(Command.CalculatePowderProjection)

    def get_powder_u1(self):
        return str(self.cmbPowderU1.currentText())

    def get_powder_u2(self):
        return str(self.cmbPowderU2.currentText())

    def set_powder_u1(self, name):
        # Signals are blocked to prevent self._u1_changed being called here (it would be false alarm)
        self.cmbPowderU1.blockSignals(True)
        self.cmbPowderU1.setCurrentIndex(self._name_to_index[name])
        self.cmbPowderU1.blockSignals(False)

    def set_powder_u2(self, name):
        # Signals are blocked to prevent self._u2_changed being called here (it would be false alarm)
        self.cmbPowderU2.blockSignals(True)
        self.cmbPowderU2.setCurrentIndex(self._name_to_index[name])
        self.cmbPowderU2.blockSignals(False)

    def populate_powder_u1(self, u1_options):
        # Signals are blocked to prevent self._u1_changed being called here (it would be false alarm)
        self.cmbPowderU1.blockSignals(True)
        self.cmbPowderU1.clear()
        # Assuming that u1 and u2 both have the same possible units.
        self._name_to_index = {}
        for idx, value in enumerate(u1_options):
            self.cmbPowderU1.addItem(value)
            self._name_to_index[value] = idx
        self.cmbPowderU1.blockSignals(False)

    def populate_powder_u2(self, u2_options):
        # Signals are blocked to prevent self._u2_changed being called here (it would be false alarm)
        self.cmbPowderU2.blockSignals(True)
        self.cmbPowderU2.clear()
        for value in u2_options:
            self.cmbPowderU2.addItem(value)
        self.cmbPowderU2.blockSignals(False)

    def populate_powder_projection_units(self, powder_projection_units):
        self.cmbPowderUnits.clear()
        for unit in powder_projection_units:
            self.cmbPowderUnits.addItem(unit)

    def get_powder_units(self):
        return str(self.cmbPowderUnits.currentText())

    def clear_displayed_error(self):
        self._display_error("")

    def _display_error(self, error_string):
        self.error_occurred.emit(error_string)
