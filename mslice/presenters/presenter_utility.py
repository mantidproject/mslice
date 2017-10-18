from .interfaces.main_presenter import MainPresenterInterface


class PresenterUtility(object):

    def register_master(self, main_presenter):
        assert (isinstance(main_presenter, MainPresenterInterface))
        self._main_presenter = main_presenter
        self._main_presenter.subscribe_to_workspace_selection_monitor(self)

    def _to_float(self, x):
        return None if x == "" else float(x)

    def _to_int(self, x):
        return None if x == "" else int(x)

    def _clear_displayed_error(self, view):
        view.clear_displayed_error()
