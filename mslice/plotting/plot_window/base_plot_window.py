from __future__ import (absolute_import, division, print_function)


class BasePlotWindow(object):
    """The Subclass MUST during construction initialize self.canvas as its canvas.

    The canvas.figure MUST be a matplotlib figure
    SLOTS
    Keep button: self._report_as_kept_to_manager
    Make Current button: self._report_as_current_to_manager
    Dump to console button : self._dump_script_to_console

    IMPLEMENT
    self._display_status """
    def __init__(self,number,manager):
        self.number = number
        self._manager = manager
        self.canvas = None

    def set_as_kept(self):
        self._display_status("kept")

    def set_as_current(self):
        self._display_status("current")

    def display_status(self,status):
        if status == "kept":
            self._display_status('kept')
        elif status == "current":
            self._display_status("current")
        else:
            raise ValueError("Invalid status %s"%status)

    def _display_status(self, status):
        pass

    def _report_as_kept_to_manager(self):
        self._manager.set_figure_as_kept(self.number)

    def _report_as_current_to_manager(self):
        self._manager.set_figure_as_current(self.number)

    def get_figure(self):
        return self.canvas.figure
