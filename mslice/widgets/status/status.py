"""A status widget

Displays information/errors to the user
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)

from qtpy.QtWidgets import QWidget

from mslice.load_ui import load_ui

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class StatusWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super(StatusWidget, self).__init__(*args, **kwargs)
        load_ui(__file__, 'status.ui', self)
