from __future__ import (absolute_import, division, print_function)
import unittest

import mock
from mock import call, patch

from mslice.models.workspacemanager.workspace_provider import WorkspaceProvider
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from mslice.views.mainview import MainView
from mslice.views.workspace_view import WorkspaceView
from mslice.widgets.workspacemanager.command import Command


#TODO Test constructor and make this test specific

class WorkspaceManagerPresenterTest(unittest.TestCase):
    def setUp(self):
        self.workspace_provider = mock.create_autospec(spec=WorkspaceProvider)
        self.view = mock.create_autospec(spec=WorkspaceView)
        self.mainview = mock.create_autospec(MainView)
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.mainview.get_presenter = mock.Mock(return_value=self.main_presenter)

    def test_register_master_success(self):
        workspace_presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        workspace_presenter.register_master(self.main_presenter)
        self.main_presenter.register_workspace_selector.assert_called_once_with(workspace_presenter)

    def test_register_master_invalid_master_fail(self):
        workspace_presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        self.assertRaises(AssertionError ,workspace_presenter.register_master, 3)

    def test_rename_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return a single selected workspace on call to get_workspace_selected and supply a
        # name on call to get_workspace_new_name
        old_workspace_name = 'file1'
        new_workspace_name = 'new_name'
        self.view.get_workspace_selected = mock.Mock(return_value=[old_workspace_name])
        self.view.get_workspace_new_name = mock.Mock(return_value=new_workspace_name)
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=['file1', 'file2', 'file3'])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_new_name.assert_called_once_with()
        self.workspace_provider.rename_workspace.assert_called_once_with(selected_workspace='file1', new_name='new_name')
        self.view.display_loaded_workspaces.assert_called_once()

    def test_rename_workspace_multiple_workspace_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports multiple selected workspaces on calls to get_workspace_selected
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_only_one_workspace.assert_called_once_with()
        self.workspace_provider.rename_workspace.assert_not_called()

    def test_rename_workspace_non_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports that no workspaces are selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        self.workspace_provider.rename_workspace.assert_not_called()

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    def test_save_workspace(self, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = r'A:\file\path'
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        save_dir_mock.return_value=(path_to_save_to, workspace_to_save, '.nxs')

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False,
                                              default_ext='.nxs')
        self.workspace_provider.save_workspace.assert_called_once_with(
            [workspace_to_save], path_to_save_to, 'file1', '.nxs')

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    def test_save_ascii_workspace(self, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = r'A:\file\path'
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        save_dir_mock.return_value = (path_to_save_to, workspace_to_save, '.txt')
        self.presenter.notify(Command.SaveSelectedWorkspaceAscii)
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False, default_ext='.txt')
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.save_workspace.assert_called_once_with(
            [workspace_to_save], path_to_save_to, 'file1', '.txt')

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    def test_save_matlab_workspace(self, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = r'A:\file\path'
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        save_dir_mock.return_value=(path_to_save_to, workspace_to_save, '.mat')

        self.presenter.notify(Command.SaveSelectedWorkspaceMatlab)
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False, default_ext='.mat')
        self.workspace_provider.save_workspace.assert_called_once_with(
            [workspace_to_save], path_to_save_to, 'file1', '.mat')

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    def test_save_workspace_multiple_selected(self, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        #Create a view that reports multiple workspaces are selected on calls to get_workspace_selected
        path_to_save_to = r'A:\file\path'
        self.view.get_workspace_selected = mock.Mock(return_value=['file1','file2'])
        save_dir_mock.return_value=(path_to_save_to, None, '.nxs')
        self.workspace_provider.save_workspace = mock.Mock(side_effect=[True,RuntimeError])

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=True, save_as_image=False, default_ext='.nxs')
        self.workspace_provider.save_workspace.assert_called_with(['file1', 'file2'], path_to_save_to, None, '.nxs')

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    def test_save_workspace_non_selected_prompt_user(self, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        #Create a view that reports no workspaces arw selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        save_dir_mock.assert_not_called()
        self.workspace_provider.save_workspace.assert_not_called()

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    def test_save_workspace_cancelled(self, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = "" # view returns empty string to indicate operation cancelled
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        save_dir_mock.return_value=(path_to_save_to, workspace_to_save, '.nxs')

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False, default_ext='.nxs')
        self.view.error_invalid_save_path.assert_called_once()
        self.workspace_provider.save_workspace.assert_not_called()

    def test_remove_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a workspace that reports a single selected workspace on calls to get_workspace_selected
        workspace_to_be_removed = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_be_removed])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.delete_workspace.assert_called_once_with(workspace_to_be_removed)
        self.view.display_loaded_workspaces.assert_called_once()

    def test_remove_multiple_workspaces(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports 3 selected workspaces on calls to get_workspace_selected
        workspace1 = 'file1'
        workspace2 = 'file2'
        workspace3 = 'file3'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace1, workspace2, workspace3])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        delete_calls = [call(workspace1), call(workspace2), call(workspace3)]
        self.workspace_provider.delete_workspace.assert_has_calls(delete_calls, any_order= True)
        self.view.display_loaded_workspaces.assert_called_once()

    def test_remove_workspace_non_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports no workspace selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()
        self.workspace_provider.delete_workspace.assert_not_called()
        self.view.display_loaded_workspaces.assert_not_called()

    def test_broadcast_success(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        self.presenter.register_master(self.main_presenter)
        self.presenter.notify(Command.SelectionChanged)
        self.main_presenter.notify_workspace_selection_changed()

    def test_call_presenter_with_unknown_command(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        unknown_command = 10
        self.assertRaises(ValueError,self.presenter.notify, unknown_command)

    def test_notify_presenter_clears_error(self):
        presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        presenter.register_master(self.main_presenter)
        # This unit test will verify that notifying cut presenter will cause the error to be cleared on the view.
        # The actual subsequent procedure will fail, however this irrelevant to this. Hence the try, except blocks
        for command in [x for x in dir(Command) if x[0] != "_"]:
            try:
                presenter.notify(command)
            except ValueError:
                pass
            self.view.clear_displayed_error.assert_called()
            self.view.reset_mock()

    def test_set_selected_workspace_index(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        self.view.get_workspace_index = mock.Mock()
        self.workspace_provider.get_workspace_name = mock.Mock()
        self.presenter.set_selected_workspaces([1])
        self.view.set_workspace_selected.assert_called_once_with([1])

    def test_set_selected_workspace_name(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        self.view.get_workspace_index = mock.Mock(return_value=0)
        self.workspace_provider.get_workspace_name = mock.Mock()
        self.presenter.set_selected_workspaces(['ws'])
        self.view.get_workspace_index.assert_called_once_with('ws')
        self.view.set_workspace_selected.assert_called_once_with([0])

    def test_set_selected_workspace_handle(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        self.view.get_workspace_index = mock.Mock(return_value=0)
        self.workspace_provider.get_workspace_name = mock.Mock(return_value='ws')
        self.presenter.set_selected_workspaces([mock.Mock()])
        self.workspace_provider.get_workspace_name.called_once_with(mock.Mock())
        self.view.get_workspace_index.assert_called_once_with('ws')
        self.view.set_workspace_selected.assert_called_once_with([0])

    def test_combine_workspace_single_ws(self):
        # Checks that it will fail if only one workspace is selected.
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        selected_workspaces = ['ws1']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.view.add_workspace_dialog = mock.Mock(return_value='ws2')
        self.presenter.notify(Command.CombineWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.add_workspace_dialog.assert_called_once()
        self.workspace_provider.combine_workspace.assert_called_once_with(['ws1', 'ws2'], 'ws1_combined')

    def test_combine_workspace_wrong_type(self):
        # Checks that it will fail if one of the workspace is not a MDEventWorkspace
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.workspace_provider.is_pixel_workspace = mock.Mock(side_effect=[True, False])
        self.presenter.notify(Command.CombineWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        check_calls = [call('ws1'), call('ws2')]
        self.workspace_provider.is_pixel_workspace.assert_has_calls(check_calls, any_order= True)
        self.view.error_select_more_than_one_workspaces.assert_called()

    def test_combine_workspace(self):
        # Now checks it suceeds otherwise
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.workspace_provider.is_pixel_workspace = mock.Mock(side_effect=[True, True])
        self.presenter.notify(Command.CombineWorkspace)
        self.view.get_workspace_selected.assert_called()
        self.view.error_select_more_than_one_workspaces.assert_not_called()
        self.workspace_provider.combine_workspace.assert_called_once_with(selected_workspaces,
                                                                          selected_workspaces[0]+'_combined')

if __name__ == '__main__':
    unittest.main()
