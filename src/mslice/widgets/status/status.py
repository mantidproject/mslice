"""A status widget

Displays information/errors to the user
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from qtpy.QtWidgets import QWidget

from mslice.util.qt import load_ui


# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class StatusWidget(QWidget):
    def __init__(self, parent=None, *args, **kwargs):
        QWidget.__init__(self, parent, *args, **kwargs)
        load_ui(__file__, 'status.ui', self)
