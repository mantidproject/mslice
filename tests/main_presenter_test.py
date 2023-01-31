from __future__ import (absolute_import, division, print_function)
import mock
import unittest

from mslice.presenters.main_presenter import MainPresenter
from mslice.presenters.interfaces.workspace_manager_presenter import WorkspaceManagerPresenterInterface
from mslice.views.interfaces.mainview import MainView

SELECTED_WORKSPACES = ['a', 'b', 'c']


class MainPresenterTests(unittest.TestCase):
    def setUp(self):
        self.mainview = mock.create_autospec(MainView)
        self.workspace_presenter = mock.create_autospec(WorkspaceManagerPresenterInterface)
        self.workspace_presenter.get_selected_workspaces = mock.Mock(return_value=SELECTED_WORKSPACES)

    def test_constructor_sucess(self):
        subpresenters = [mock.Mock() for i in range(10)]
        main_presenter = MainPresenter(self.mainview, *subpresenters)
        for presenter in subpresenters:
            presenter.register_master.assert_called_once_with(main_presenter)

    def test_get_selected_workspaces_success(self):
        main_presenter = MainPresenter(self.mainview)
        main_presenter.register_workspace_selector(self.workspace_presenter)
        return_value = main_presenter.get_selected_workspaces()
        self.workspace_presenter.get_selected_workspaces.assert_called()
        self.assertTrue(return_value == SELECTED_WORKSPACES)

    def test_selection_change_broadcast(self):
        main_presenter = MainPresenter(self.mainview)
        clients = [mock.Mock(), mock.Mock(), mock.Mock()]
        for client in clients:
            main_presenter.subscribe_to_workspace_selection_monitor(client)

        for client in clients:
            client.workspace_selection_changed.assert_not_called()

        main_presenter.notify_workspace_selection_changed()
        for client in clients:
            client.workspace_selection_changed.assert_called_once()

    def test_subscribe_invalid_listener_fail(self):
        main_presenter = MainPresenter(self.mainview)
        class x:
            def __init__(self):
                self.attr = 1
        self.assertRaises(TypeError,main_presenter.subscribe_to_workspace_selection_monitor, x())

    def test_subscribe_invalid_listener_non_callable_handle_fail(self):
        main_presenter = MainPresenter(self.mainview)
        class x:
            def __init__(self):
                self.workspace_selection_changed = 1
        self.assertRaises(TypeError, main_presenter.subscribe_to_workspace_selection_monitor, x())

    def test_update_displayed_workspaces(self):
        main_presenter = MainPresenter(self.mainview)
        main_presenter.register_workspace_selector(self.workspace_presenter)
        main_presenter.update_displayed_workspaces()
        self.workspace_presenter.update_displayed_workspaces.assert_called_with()

    def test_set_energy_default(self):
        main_presenter = MainPresenter(self.mainview)
        clients = [[mock.Mock(), True], [mock.Mock(), False], [mock.Mock(), True]]
        for client, enable in clients:
            if not enable:
                client.set_energy_default = mock.NonCallableMagicMock()
            main_presenter.subscribe_to_energy_default_monitor(client)

        for client, enable in clients:
            client.set_energy_default.assert_not_called()

        main_presenter.set_energy_default('meV')
        for client, enable in clients:
            if enable:
                client.set_energy_default.assert_called_once()
            else:
                client.set_energy_default.assert_not_called()

    def test_energy_conversion_options(self):
        main_presenter = MainPresenter(self.mainview)
        main_presenter.is_energy_conversion_allowed()
        self.mainview.is_energy_conversion_allowed.assert_called()

    def test_set_cut_algorithm_default(self):
        main_presenter = MainPresenter(self.mainview)
        clients = [[mock.Mock(), True], [mock.Mock(), False], [mock.Mock(), True]]
        for client, enable in clients:
            if not enable:
                client.set_cut_algorithm_default = mock.NonCallableMagicMock()
            main_presenter.subscribe_to_cut_algo_default_monitor(client)

        for client, enable in clients:
            client.set_energy_default.assert_not_called()

        main_presenter.set_cut_algorithm_default('Rebin')
        for client, enable in clients:
            if enable:
                client.set_cut_algorithm_default.assert_called_once()
            else:
                client.set_cut_algorithm_default.assert_not_called()
