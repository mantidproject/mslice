from __future__ import (absolute_import, division, print_function)

from mslice.views.interfaces.powder_projection_view import PowderView


class CLIPowderWidget(PowderView):
    """
    A Powder Widget dummy class to interface with the presenter classes without doing anything in CLI mode.
    """

    def __init__(self):
        pass

    def get_presenter(self):
        pass

    def _u1_changed(self):
        pass

    def _u2_changed(self):
        pass

    def _btn_clicked(self):
        pass

    def get_powder_u1(self):
        pass

    def get_powder_u2(self):
        pass

    def set_powder_u1(self, name):
        pass

    def set_powder_u2(self, name):
        pass

    def populate_powder_u1(self, u1_options):
        pass

    def populate_powder_u2(self, u2_options):
        pass

    def populate_powder_projection_units(self, powder_projection_units):
        pass

    def get_powder_units(self):
        pass

    def disable_calculate_projections(self, disable):
        pass

    def display_projection_error(self, message):
        pass

    def clear_displayed_error(self):
        pass

    def _display_error(self, error_string):
        pass

    def display_message_box(self, message):
        pass
