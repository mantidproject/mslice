"""A widget for projection calculations for single crystal
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from PyQt4.QtGui import QWidget

from singlecrystal_ui import Ui_Form

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class SingleCrystalWidget(QWidget, Ui_Form):
    def __init__(self, *args, **kwargs):
        super(SingleCrystalWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
