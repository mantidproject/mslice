from command import Command
from SlicePlotter import SlicePlotter


class SlicePlotterPresenter:
    def __init__(self, main_view, slice_view):
        self._slice_view = slice_view
        self._main_presenter = main_view.get_presenter()

    def notify(self,command):
        if command == Command.DisplaySlice:
            selected_workspaces = self._main_presenter.get_selected_workspaces()
            if not selected_workspaces:
                self._slice_view.error_select_one_workspace()
                return
            if len(selected_workspaces) > 1:
                pass
                #TODO is this okay? plot multiple? or error?
            selected_workspace = selected_workspaces[0]
