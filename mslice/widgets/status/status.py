"""A status widget

Displays information/errors to the user
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)

from PyQt4.QtGui import QWidget

from .status_ui import Ui_Form

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class StatusWidget(QWidget, Ui_Form):
    def __init__(self, *args, **kwargs):
        super(StatusWidget, self).__init__(*args, **kwargs)
        self.setupUi(self)
