import mock
import unittest
from presenters.main_presenter import MainPresenter
from presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from mainview import MainView
SELECTED_WORKSPACES = ['a', 'b', 'c']


class MainPresenterTests(unittest.TestCase):
    def setUp(self):
        self.mainview = mock.create_autospec(MainView)
        self.workspace_presenter = mock.create_autospec(WorkspaceManagerPresenter)
        self.workspace_presenter.get_selected_workspaces = mock.Mock(return_value=SELECTED_WORKSPACES)

    def test_get_selected_workspaces_success(self):
        main_presenter = MainPresenter(self.mainview, self.workspace_presenter)
        return_value = main_presenter.get_selected_workspaces()
        self.workspace_presenter.get_selected_workspaces.assert_called()
        self.assert_(return_value == SELECTED_WORKSPACES)

    def test_selection_change_broadcast(self):
        main_presenter = MainPresenter(self.mainview, self.workspace_presenter)
        clients = [mock.Mock(), mock.Mock(), mock.Mock()]
        for client in clients:
            main_presenter.subscribe_to_workspace_selection_monitor(client)

        for client in clients:
            client.workspace_selection_changed.assert_not_called()

        main_presenter.broadcast_selection_changed()
        for client in clients:
            client.workspace_selection_changed.assert_called_once()

    def test_subsribe_invalid_listener_fail(self):
        main_presenter = MainPresenter(self.mainview, self.workspace_presenter)
        class x:
            def __init__(self):
                self.attr = 1
        self.assertRaises(TypeError,main_presenter.subscribe_to_workspace_selection_monitor, x())

    def test_subscribe_invalid_listener_non_callable_handle_fail(self):
        main_presenter = MainPresenter(self.mainview, self.workspace_presenter)
        class x:
            def __init__(self):
                self.workspace_selection_changed = 1
        self.assertRaises(TypeError,main_presenter.subscribe_to_workspace_selection_monitor, x())

    def test_update_displayed_workspaces(self):
        main_presenter = MainPresenter(self.mainview, self.workspace_presenter)
        main_presenter.update_displayed_workspaces()
        self.workspace_presenter.update_displayed_workspaces.assert_called_with()
