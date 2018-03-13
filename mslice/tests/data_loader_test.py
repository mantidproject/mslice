from __future__ import (absolute_import, division, print_function)
from tempfile import gettempdir
from os.path import join
import unittest

import mock
from mock import call, patch

from mslice.models.workspacemanager.workspace_provider import WorkspaceProvider
from mslice.presenters.data_loader_presenter import DataLoaderPresenter
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.widgets.dataloader.dataloader import DataLoaderWidget

class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        self.workspace_provider = mock.create_autospec(spec=WorkspaceProvider)
        self.view = mock.create_autospec(spec=DataLoaderWidget)
        self.view.busy.emit = mock.Mock()
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.presenter = DataLoaderPresenter(self.view)
        self.presenter.register_master(self.main_presenter)
        self.presenter.set_workspace_provider(self.workspace_provider)

    def test_load_one_workspace(self):
        # Create a view that will return a path on call to get_workspace_to_load_path
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path_to_nexus = join(tempdir, 'cde.nxs')
        workspace_name = 'cde'
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=[])
        self.workspace_provider.get_EMode = mock.Mock(return_value='Indirect')
        self.workspace_provider.has_efixed = mock.Mock(return_value=False)
        self.workspace_provider.set_efixed = mock.Mock()
        self.view.get_workspace_efixed = mock.Mock(return_value=(1.845, False))

        self.presenter.load_workspace([path_to_nexus])
        self.workspace_provider.load.assert_called_with(filename=path_to_nexus, output_workspace=workspace_name)
        self.view.get_workspace_efixed.assert_called_with(workspace_name, False)
        self.workspace_provider.set_efixed.assert_called_once_with(workspace_name, 1.845)
        self.main_presenter.show_workspace_manager_tab.assert_called_once()
        self.main_presenter.show_tab_for_workspace.assert_called_once()
        self.main_presenter.update_displayed_workspaces.assert_called_once()

    def test_load_multiple_workspaces(self):
        # Create a view that will return three filepaths on 3 subsequent calls to get_workspace_to_load_path
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path1 = join(tempdir, 'file1.nxs')
        path2 = join(tempdir, 'file2.nxs')
        path3 = join(tempdir, 'file3.nxs')
        ws_name1 = 'file1'
        ws_name2 = 'file2'
        ws_name3 = 'file3'
        # Make the third workspace something not in current workspace list, so don't need ask overwrite
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=[ws_name1, ws_name2, ''])
        self.workspace_provider.get_EMode = mock.Mock(return_value='Direct')
        # Makes the first file not load because of a name collision
        self.view.confirm_overwrite_workspace = mock.Mock(side_effect=[False, True, True])
        # Makes the second file fail to load, to check if it raise the correct error
        self.workspace_provider.load = mock.Mock(side_effect=[RuntimeError, 0, 0])
        self.presenter.load_workspace([path1, path2, path3])
        # Because of the name collision, the first file name is not loaded.
        load_calls = [call(filename=path2, output_workspace=ws_name2),
                      call(filename=path3, output_workspace=ws_name3)]
        self.workspace_provider.load.assert_has_calls(load_calls)
        self.view.error_unable_to_open_file.assert_called_once_with(ws_name2)
        self.view.no_workspace_has_been_loaded.assert_called_once_with(ws_name1)
        self.view.get_workspace_efixed.assert_not_called()

    def test_load_workspace_dont_overwrite(self):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path = join(tempdir, 'file.nxs')
        ws_name = 'file'
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=[ws_name])
        self.workspace_provider.get_EMode = mock.Mock(return_value='Direct')
        self.view.confirm_overwrite_workspace = mock.Mock(return_value=False)

        self.presenter.load_workspace([path])
        self.view.confirm_overwrite_workspace.assert_called_once()
        self.view.no_workspace_has_been_loaded.assert_called_once()

    def test_load_workspace_fail(self):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path_to_nexus = join(tempdir, 'cde.nxs')
        workspace_name = 'cde'
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=[workspace_name])
        self.workspace_provider.load = mock.Mock(side_effect=RuntimeError)

        self.presenter.load_workspace([path_to_nexus])
        self.view.error_unable_to_open_file.assert_called_once()

    def test_load_and_merge(self):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path1 = join(tempdir, 'file1.nxs')
        path2 = join(tempdir, 'file2.nxs')
        path3 = join(tempdir, 'file3.nxs')
        self.workspace_provider.get_EMode = mock.Mock(return_value='Direct')

        self.presenter.load_workspace([path1, path2, path3], True)
        self.workspace_provider.load.assert_called_once_with(filename=path1 + '+' + path2 +'+' + path3,
                                                             output_workspace='file1_merged')

    @patch('mslice.presenters.data_loader_presenter.load_from_ascii')
    def test_load_ascii(self, load_ascii_mock):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path = join(tempdir, 'abc.txt')
        self.workspace_provider.get_EMode = mock.Mock(return_value='Direct')

        self.presenter.load_workspace([path])
        load_ascii_mock.assert_called_once_with(path, 'abc')
        self.workspace_provider.load.assert_not_called()
        self.view.get_workspace_efixed.assert_not_called()
        self.main_presenter.show_workspace_manager_tab.assert_called_once()
        self.main_presenter.show_tab_for_workspace.assert_called_once()
        self.main_presenter.update_displayed_workspaces.assert_called_once()

    @patch('mslice.presenters.data_loader_presenter.load_from_ascii')
    def test_load_multiple_types(self, load_ascii_mock):
        tempdir = gettempdir()  # To ensure sample paths are valid on platform of execution
        path_ascii = join(tempdir, 'ascii.txt')
        path_nexus = join(tempdir, 'nexus.nxs')
        self.workspace_provider.get_EMode = mock.Mock(return_value='Direct')
        self.presenter.load_workspace([path_ascii, path_nexus])
        self.workspace_provider.load.assert_called_once_with(filename=path_nexus, output_workspace='nexus')
        load_ascii_mock.assert_called_once_with(path_ascii, 'ascii')
        self.main_presenter.show_workspace_manager_tab.assert_called()
        self.main_presenter.update_displayed_workspaces.assert_called_once()
