import unittest

import mock
from mock import call

from models.workspacemanager.workspace_provider import WorkspaceProvider
from presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from presenters.interfaces.main_presenter import MainPresenterInterface
from views.workspace_view import WorkspaceView
from widgets.workspacemanager.command import Command
from mainview import MainView
from tempfile import gettempdir
from os.path import join

#TODO handle mantid load fail inquiry
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

    def test_load_one_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return a path on call to get_workspace_to_load_path
        tempdir = gettempdir()  # To insure sample paths are valid on platform of execution
        path_to_nexus = join(tempdir,'cde.nxs')
        workspace_name = 'cde'
        self.view.get_workspace_to_load_path = mock.Mock(return_value=[path_to_nexus])
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=[workspace_name])
        self.view.get_workspace_index = mock.Mock(return_value=0)

        self.presenter.notify(Command.LoadWorkspace)
        self.view.get_workspace_to_load_path.assert_called_once()
        self.workspace_provider.load.assert_called_with(filename=path_to_nexus, output_workspace=workspace_name)
        self.view.display_loaded_workspaces.assert_called_with([workspace_name])
        self.view.set_workspace_selected.assert_called_with([0])

    def test_load_multiple_workspaces(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return three filepaths on on 3 subsequent calls to get_workspace_to_load_path
        tempdir = gettempdir()  # To insure sample paths are valid on platform of execution
        path1 = join(tempdir,'file1.nxs')
        path2 = join(tempdir,'file2.nxs')
        path3 = join(tempdir,'file3.nxs')
        ws_name1 = 'file1'
        ws_name2 = 'file2'
        ws_name3 = 'file3'
        self.view.get_workspace_to_load_path = mock.Mock(
            return_value=[path1, path2, path3])
        self.workspace_provider.get_workspace_names = mock.Mock(
            return_value=[ws_name1, ws_name2, ws_name3])
        self.presenter.notify(Command.LoadWorkspace)
        load_calls = [call(filename=path1, output_workspace=ws_name1),
                      call(filename=path2, output_workspace=ws_name2),
                      call(filename=path3, output_workspace=ws_name3)]
        self.workspace_provider.load.assert_has_calls(load_calls)

    def test_load_workspace_cancelled(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return a path on call to get_workspace_to_load_path
        # This assumes the view will return an empty string if the operation was cancelled
        self.view.get_workspace_to_load_path = mock.Mock(return_value='')

        self.presenter.notify(Command.LoadWorkspace)
        self.view.get_workspace_to_load_path.assert_called_once()
        self.workspace_provider.load.assert_not_called()

    def test_load_workspace_dont_overwrite(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        tempdir = gettempdir()  # To insure sample paths are valid on platform of execution
        path = join(tempdir,'file.nxs')
        ws_name = 'file'
        self.view.get_workspace_to_load_path = mock.Mock(return_value=[path])
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=[ws_name])
        self.view.confirm_overwrite_workspace = mock.Mock(return_value=False)

        self.presenter.notify(Command.LoadWorkspace)
        self.view.confirm_overwrite_workspace.assert_called_once()

    def test_load_workspace_fail(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return a path on call to get_workspace_to_load_path
        tempdir = gettempdir()  # To insure sample paths are valid on platform of execution
        path_to_nexus = join(tempdir,'cde.nxs')
        workspace_name = 'cde'
        self.view.get_workspace_to_load_path = mock.Mock(return_value=path_to_nexus)
        self.workspace_provider.get_workspace_names = mock.Mock(return_value=[workspace_name])
        self.workspace_provider.load = mock.Mock(side_effect=RuntimeError)

        self.presenter.notify(Command.LoadWorkspace)
        self.view.get_workspace_to_load_path.assert_called_once()
        self.view.error_unable_to_open_file.assert_called_once()


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

    def test_save_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = r'A:\file\path\save.nxs'
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        self.view.get_workspace_to_save_filepath = mock.Mock(return_value=path_to_save_to)

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_called_once_with()
        self.workspace_provider.save_nexus.assert_called_once_with(workspace_to_save, path_to_save_to)

    def test_save_workspace_multiple_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        #Create a view that reports multiple workspaces are selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=['file1','file2'])

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_only_one_workspace.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_not_called()
        self.workspace_provider.save_nexus.assert_not_called()

    def test_save_workspace_non_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        #Create a view that reports no workspaces arw selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_not_called()
        self.workspace_provider.save_nexus.assert_not_called()

    def test_save_workspace_cancelled(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = "" # view returns empty string to indicate operation cancelled
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        self.view.get_workspace_to_save_filepath = mock.Mock(return_value=path_to_save_to)

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_called_once_with()
        self.view.error_invalid_save_path.assert_called_once()
        self.workspace_provider.save_nexus.assert_not_called()

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
        for command in filter(lambda x: x[0] != "_", dir(Command)):
            try:
                presenter.notify(command)
            except:
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

if __name__ == '__main__':
    unittest.main()
