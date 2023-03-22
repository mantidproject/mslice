from __future__ import (absolute_import, division, print_function)
from tempfile import gettempdir
from os.path import join
import unittest

import mock
from mock import call, patch, PropertyMock

from mslice.presenters.data_loader_presenter import DataLoaderPresenter
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.widgets.dataloader.dataloader import DataLoaderWidget
from mslice.workspace.workspace import Workspace


class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        self.view = mock.create_autospec(spec=DataLoaderWidget)
        self.view.busy.emit = mock.Mock()
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.presenter = DataLoaderPresenter(self.view)
        self.presenter.register_master(self.main_presenter)

    @patch('mslice.models.workspacemanager.workspace_algorithms.process_limits')
    @patch('mslice.presenters.data_loader_presenter.load')
    @patch('mslice.presenters.data_loader_presenter.get_workspace_handle')
    def test_load_one_workspace(self, get_ws_handle_mock, load_mock, process_limits):
        # Create a view that will return a path on call to get_workspace_to_load_path
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path_to_nexus = join(tempdir, 'cde.nxs')
        workspace_name = 'cde'
        self.view.get_workspace_efixed = mock.Mock(return_value=(1.845, False))
        ws_mock = mock.Mock(spec=Workspace)
        get_ws_handle_mock.return_value = ws_mock
        e_fixed = PropertyMock()
        e_mode = PropertyMock(return_value="Indirect")
        ef_defined = PropertyMock(return_value=False)
        type(ws_mock).e_fixed = e_fixed
        type(ws_mock).e_mode = e_mode
        type(ws_mock).ef_defined = ef_defined

        with patch('mslice.models.workspacemanager.workspace_algorithms.get_workspace_handle') as gwh:
            gwh.return_value = ws_mock
            limits = PropertyMock(side_effect=({} if i < 2 else {'DeltaE': [-1, 1]} for i in range(6)))
            type(ws_mock).limits = limits
            e_fixed.return_value = 1.845
            self.presenter.load_workspace([path_to_nexus])
        load_mock.assert_called_with(filename=path_to_nexus, output_workspace=workspace_name)
        e_fixed.assert_has_calls([call(1.845), call()])
        process_limits.assert_called_once_with(ws_mock)
        self.main_presenter.show_workspace_manager_tab.assert_called_once()
        self.main_presenter.show_tab_for_workspace.assert_called_once()
        self.main_presenter.update_displayed_workspaces.assert_called_once()

    @patch('mslice.presenters.data_loader_presenter.load')
    @patch('mslice.presenters.data_loader_presenter.get_visible_workspace_names')
    @patch('mslice.presenters.data_loader_presenter.get_workspace_handle')
    def test_load_multiple_workspaces(self, get_ws_handle_mock, get_ws_names_mock, load_mock):
        # Create a view that will return three filepaths on 3 subsequent calls to get_workspace_to_load_path
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path1 = join(tempdir, 'file1.nxs')
        path2 = join(tempdir, 'file2.nxs')
        path3 = join(tempdir, 'file3.nxs')
        ws_name1 = 'file1'
        ws_name2 = 'file2'
        ws_name3 = 'file3'
        # Make the third workspace something not in current workspace list, so don't need ask overwrite
        get_ws_names_mock.return_value = [ws_name1, ws_name2, '']
        get_ws_handle_mock.e_mode.return_value = 'Direct'
        # Makes the first file not load because of a name collision
        self.view.confirm_overwrite_workspace = mock.Mock(side_effect=[False, True, True])
        # Makes the second file fail to load, to check if it raise the correct error
        load_mock.side_effect=[RuntimeError, 0, 0]
        self.presenter.load_workspace([path1, path2, path3])
        # Because of the name collision, the first file name is not loaded.
        load_calls = [call(filename=path2, output_workspace=ws_name2),
                      call(filename=path3, output_workspace=ws_name3)]
        load_mock.assert_has_calls(load_calls)
        self.view.error_unable_to_open_file.assert_called_once_with(ws_name2)
        self.view.no_workspace_has_been_loaded.assert_called_once_with(ws_name1)
        self.view.get_workspace_efixed.assert_not_called()

    @patch('mslice.presenters.data_loader_presenter.get_visible_workspace_names')
    @patch('mslice.presenters.data_loader_presenter.get_workspace_handle')
    def test_load_workspace_dont_overwrite(self, get_ws_handle_mock, get_ws_names_mock):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path = join(tempdir, 'file.nxs')
        ws_name = 'file'
        get_ws_names_mock.return_value = [ws_name]
        get_ws_handle_mock.e_mode.return_value = 'Direct'
        self.view.confirm_overwrite_workspace = mock.Mock(return_value=False)

        self.presenter.load_workspace([path])
        self.view.confirm_overwrite_workspace.assert_called_once()
        self.view.no_workspace_has_been_loaded.assert_called_once()

    @patch('mslice.presenters.data_loader_presenter.load')
    @patch('mslice.presenters.data_loader_presenter.get_visible_workspace_names')
    def test_load_workspace_fail(self, get_ws_names_mock, load_mock):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path_to_nexus = join(tempdir, 'cde.nxs')
        workspace_name = 'cde'
        get_ws_names_mock.return_value=[workspace_name]
        load_mock.side_effect=RuntimeError

        self.presenter.load_workspace([path_to_nexus])
        self.view.error_unable_to_open_file.assert_called_once()

    @patch('mslice.presenters.data_loader_presenter.load')
    @patch('mslice.presenters.data_loader_presenter.get_workspace_handle')
    def test_load_and_merge(self, get_ws_handle_mock, load_mock):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path1 = join(tempdir, 'file1.nxs')
        path2 = join(tempdir, 'file2.nxs')
        path3 = join(tempdir, 'file3.nxs')
        # self.workspace_provider.get_EMode = mock.Mock(return_value='Direct')
        ws_mock = mock.Mock()
        get_ws_handle_mock.return_value = ws_mock
        self.presenter.load_workspace([path1, path2, path3], True)
        load_mock.assert_called_once_with(filename=path1 + '+' + path2 + '+' + path3,
                                          output_workspace='file1_merged')

    @patch('mslice.presenters.data_loader_presenter.load_from_ascii')
    @patch('mslice.presenters.data_loader_presenter.get_workspace_handle')
    @patch('mslice.presenters.data_loader_presenter.load')
    def test_load_ascii(self, load_mock, get_ws_handle_mock, load_ascii_mock):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path = join(tempdir, 'abc.txt')
        get_ws_handle_mock.return_value.e_mode = 'Direct'

        self.presenter.load_workspace([path])
        load_ascii_mock.assert_called_once_with(path, 'abc')
        load_mock.assert_not_called()
        self.view.get_workspace_efixed.assert_not_called()
        self.main_presenter.show_workspace_manager_tab.assert_called_once()
        self.main_presenter.show_tab_for_workspace.assert_called_once()
        self.main_presenter.update_displayed_workspaces.assert_called_once()

    @patch('mslice.presenters.data_loader_presenter.load_from_ascii')
    @patch('mslice.presenters.data_loader_presenter.get_workspace_handle')
    @patch('mslice.presenters.data_loader_presenter.load')
    def test_load_multiple_types(self, load_mock, get_ws_handle_mock, load_ascii_mock):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path_ascii = join(tempdir, 'ascii.txt')
        path_nexus = join(tempdir, 'nexus.nxs')
        get_ws_handle_mock.return_value.e_mode = 'Direct'

        self.presenter.load_workspace([path_ascii, path_nexus])
        load_mock.assert_called_once_with(filename=path_nexus, output_workspace='nexus')
        load_ascii_mock.assert_called_once_with(path_ascii, 'ascii')
        self.main_presenter.show_workspace_manager_tab.assert_called()
        self.main_presenter.update_displayed_workspaces.assert_called()
