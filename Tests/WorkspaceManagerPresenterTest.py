from widgets.workspacemanager.workspace_manager_presenter import WorkspaceManagerPresenter
from widgets.workspacemanager.workspace_provider import WorkspaceProvider
from widgets.workspacemanager.workspace_view import WorkspaceView
from widgets.workspacemanager.command import Command
import unittest
import mock
from mock import call


#TODO handle mantid load fail inquiry
#TODO Test constructor and make this test specific

class WorkspaceManagerPresenterTest(unittest.TestCase):
    def setUp(self):
        self.workspace_provider = mock.create_autospec(spec=WorkspaceProvider)
        self.view = mock.create_autospec(spec=WorkspaceView)

    def test_load_one_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return a path on call to get_workspace_to_load_path
        path_to_nexus = r'C:\a\b\cde.nxs'
        workspace_name = 'cde'
        self.view.get_workspace_to_load_path = mock.Mock(return_value=path_to_nexus)
        self.workspace_provider.getWorkspaceNames = mock.Mock(return_value=[workspace_name])

        self.presenter.notify(Command.LoadWorkspace)
        self.view.get_workspace_to_load_path.assert_called_once()
        self.workspace_provider.Load.assert_called_with(Filename=path_to_nexus, OutputWorkspace=workspace_name)
        self.view.display_loaded_workspaces.assert_called_with([workspace_name])

    def test_load_multiple_workspaces(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return three filepaths on on 3 subsequent calls to get_workspace_to_load_path
        path1 = r'C:\path\to\file1.nxs'
        path2 = r'C:\path\to\any\file2.nxs'
        path3 = r'C:\path\to\file3.nxs'
        ws_name1 = 'file1'
        ws_name2 = 'file2'
        ws_name3 = 'file3'
        self.view.get_workspace_to_load_path = mock.Mock(
            side_effect=[path1, path2, path3])
        self.workspace_provider.getWorkspaceNames = mock.Mock(side_effect=[[ws_name1], [ws_name1,ws_name2],
                                                                           [ws_name1, ws_name2, ws_name3]])
        for i in range(3):
            self.presenter.notify(Command.LoadWorkspace)
        load_calls = [call(Filename=path1, OutputWorkspace=ws_name1),
                      call(Filename=path2, OutputWorkspace=ws_name2),
                      call(Filename=path3, OutputWorkspace=ws_name3)]
        self.workspace_provider.Load.assert_has_calls(load_calls)
        display_workspace_calls = [call([ws_name1]), call([ws_name1, ws_name2]), call([ws_name1, ws_name2, ws_name3])]
        self.view.display_loaded_workspaces.assert_has_calls(display_workspace_calls)

    def test_rename_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that will return a single selected workspace on call to get_workspace_selected and supply a
        # name on call to get_workspace_new_name
        old_workspace_name = 'file1'
        new_workspace_name = 'new_name'
        self.view.get_workspace_selected = mock.Mock(return_value=[old_workspace_name])
        self.view.get_workspace_new_name = mock.Mock(return_value=new_workspace_name)
        self.workspace_provider.getWorkspaceNames = mock.Mock(return_value=['file1', 'file2', 'file3'])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_new_name.assert_called_once_with()
        self.workspace_provider.RenameWorkspace.assert_called_once_with(selected_workspace='file1', newName='new_name')
        self.view.display_loaded_workspaces.assert_called_once()

    def test_rename_workspace_multiple_workspace_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports multiple selected workspaces on calls to get_workspace_selected
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_only_one_workspace.assert_called_once_with()
        self.workspace_provider.RenameWorkspace.assert_not_called()

    def test_rename_workspace_non_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports that no workspaces are selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        self.workspace_provider.RenameWorkspace.assert_not_called()

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
        self.workspace_provider.SaveNexus.assert_called_once_with(workspace_to_save, path_to_save_to)

    def test_save_workspace_multiple_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        #Create a view that reports multiple workspaces are selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=['file1','file2'])

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_only_one_workspace.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_not_called()
        self.workspace_provider.SaveNexus.assert_not_called()

    def test_save_workspace_non_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        #Create a view that reports no workspaces arw selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_not_called()
        self.workspace_provider.SaveNexus.assert_not_called()

    def test_group_workspaces(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create view that reports two workspaces are selected on calls two get_workspace_selected
        workspaces_to_group = ['file1', 'file2']
        self.view.get_workspace_selected = mock.Mock(return_value=workspaces_to_group)

        self.presenter.notify(Command.GroupSelectedWorkSpaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.GroupWorkspaces.assert_called_with(workspaces_to_group, 'group1')
        self.view.display_loaded_workspaces.assert_called_once()

    def test_group_single_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports a single selected workspace on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=['file1'])

        self.presenter.notify(Command.GroupSelectedWorkSpaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.GroupWorkspaces.assert_called_with(['file1'], 'group1')
        self.view.display_loaded_workspaces.assert_called_once()

    def test_group_workspaces_non_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports no workspace selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.GroupSelectedWorkSpaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()
        self.workspace_provider.GroupWorkspaces.assert_not_called()
        self.view.display_loaded_workspaces.assert_not_called()

    def test_remove_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a workspace that reports a single selected workspace on calls to get_workspace_selected
        workspace_to_be_removed = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_be_removed])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.DeleteWorkspace.assert_called_once_with(workspace_to_be_removed)
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
        self.workspace_provider.DeleteWorkspace.assert_has_calls(delete_calls,any_order= True)
        self.view.display_loaded_workspaces.assert_called_once()

    def test_remove_workspace_non_selected_prompt_user(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        # Create a view that reports no workspace selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()
        self.workspace_provider.DeleteWorkspace.assert_not_called()
        self.view.display_loaded_workspaces.assert_not_called()

    def test_call_presenter_with_unknown_command(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        unknown_command = 10
        self.assertRaises(ValueError,self.presenter.notify, unknown_command)


if __name__ == '__main__':
    unittest.main()
