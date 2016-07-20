from mock.mock import self

from WorkspaceManagerPresenter import WorkspaceManagerPresenter
from WorkspaceProvider import WorkspaceProvider
from WorkspaceView import WorkspaceView
from command import Command
import unittest
import mock
from mock import call


#ToDo askOwen, view related tests, i.e view returns 'selected_workspace' that does not exist
#TODO handle mantid load fail inquiry
        #TODO Test constructor and make this test specific

class WorkspaceManagerPresenterTest(unittest.TestCase):
    def setUp(self):
        self.workspace_provider = mock.create_autospec(spec=WorkspaceProvider)
        self.view = mock.create_autospec(spec=WorkspaceView)
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)


    def test_load_one_workspace(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        self.view.get_workspace_to_load_path = mock.Mock(return_value=r'C:\a\b\cde.nxs')
        self.workspace_provider.getWorkspaceNames = mock.Mock(return_value=['cde'])
        self.presenter.notify(Command.LoadWorkspace)
        self.view.get_workspace_to_load_path.assert_called_once()
        self.workspace_provider.Load.assert_called_with(Filename=r'C:\a\b\cde.nxs', OutputWorkspace='cde')
        self.view.display_loaded_workspaces.assert_called_with(['cde']) #TODO could this be has is_called?

    def test_load_multiple_workspaces(self):
        self.presenter = WorkspaceManagerPresenter(self.view, self.workspace_provider)
        self.view.get_workspace_to_load_path = mock.Mock(
            side_effect=[r'C:\path\to\file1.nxs', r'C:\path\to\any\file2.nxs', r'C:\path\to\file3.nxs'])
        self.workspace_provider.getWorkspaceNames = mock.Mock(side_effect=[['file1'], ['file1', 'file2'],
                                                                           ['file1', 'file2', 'file3']])
        for i in range(3):
            self.presenter.notify(Command.LoadWorkspace)
        load_calls = [call(Filename='C:\\path\\to\\file1.nxs', OutputWorkspace='file1'),
                      call(Filename='C:\\path\\to\\any\\file2.nxs', OutputWorkspace='file2'),
                      call(Filename='C:\\path\\to\\file3.nxs', OutputWorkspace='file3')]
        self.workspace_provider.Load.assert_has_calls(load_calls)
        #The test should just finish here
        # <Not Maybe not useful> TODO askOwen case1
        display_workspace_calls = [call(['file1']), call(['file1', 'file2']), call(['file1', 'file2', 'file3'])]
        self.view.display_loaded_workspaces.assert_has_calls(display_workspace_calls)
        # </Maybe not useful>

    # TODO make sure that workspace provider is not called on failure cases
    def test_rename_workspace(self):
        self.view.get_workspace_selected = mock.Mock(return_value=['file1'])
        self.view.get_workspace_new_name = mock.Mock(return_value='new_name')
        self.workspace_provider.getWorkspaceNames = mock.Mock(return_value=['file1','file2','file3'])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_new_name.assert_called_once_with()
        self.workspace_provider.RenameWorkspace.assert_called_once_with(selected_workspace='file1',newName='new_name')
        print self.view.display_loaded_workspaces.assert_called_once()


    def test_rename_workspace_multiple_workspace_selected_prompt_user(self):
        self.view.get_workspace_selected = mock.Mock(return_value= ['ws1','ws2'])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_only_one_workspace.assert_called_once_with()
        self.workspace_provider.RenameWorkspace.assert_not_called() #TODO assert object not touched

    def test_rename_workspace_non_selected_prompt_user(self):
        self.view.get_workspace_selected = mock.Mock(return_value= [])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        self.workspace_provider.RenameWorkspace.assert_not_called()

    def test_save_workspace(self):
        self.view.get_workspace_selected = mock.Mock(return_value= ['file1'])
        self.view.get_workspace_to_save_filepath = mock.Mock(return_value=r'A:\file\path\save.nxs')

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_called_once_with()
        self.workspace_provider.SaveNexus.assert_called_once_with('file1',r'A:\file\path\save.nxs')


    def test_save_workspace_multiple_selected_prompt_user(self):
        self.view.get_workspace_selected = mock.Mock(return_value= ['file1','file2'])

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_only_one_workspace.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_not_called()
        self.workspace_provider.SaveNexus.assert_not_called()

    def test_save_workspace_non_selected_prompt_user(self):
        self.view.get_workspace_selected = mock.Mock(return_value= [])

        self.presenter.notify(Command.SaveSelectedWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        self.view.get_workspace_to_save_filepath.assert_not_called()
        self.workspace_provider.SaveNexus.assert_not_called()

    def test_group_workspaces(self):
        self.view.get_workspace_selected = mock.Mock(return_value= ['file1','file2'])

        self.presenter.notify(Command.GroupSelectedWorkSpaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.GroupWorkspaces.assert_called_with(['file1','file2'],'group1')
        print self.view.display_loaded_workspaces.assert_called_once()

    def test_group_single_workspace(self):
        self.view.get_workspace_selected = mock.Mock(return_value= ['file1'])

        self.presenter.notify(Command.GroupSelectedWorkSpaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.GroupWorkspaces.assert_called_with(['file1'],'group1')
        self.view.display_loaded_workspaces.assert_called_once()

    def test_group_workspaces_non_selected_prompt_user(self):
        self.view.get_workspace_selected = mock.Mock(return_value= [])

        self.presenter.notify(Command.GroupSelectedWorkSpaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()
        self.workspace_provider.GroupWorkspaces.assert_not_called()
        self.view.display_loaded_workspaces.assert_not_called()

    def test_remove_workspace(self):
        self.view.get_workspace_selected = mock.Mock(return_value= ['file1'])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.workspace_provider.DeleteWorkspace.assert_called_once_with('file1')
        self.view.display_loaded_workspaces.assert_called_once()

    def test_remove_multiple_workspaces(self):
        self.view.get_workspace_selected = mock.Mock(return_value=['file1','file2','file3'])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        deleteCalls = [call('file1'),call('file2'),call('file3')]
        self.workspace_provider.DeleteWorkspace.assert_has_calls(deleteCalls,any_order= True)
        self.view.display_loaded_workspaces.assert_called_once()

    def test_remove_workspace_non_selected_prompt_user(self):
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()
        self.workspace_provider.DeleteWorkspace.assert_not_called()
        self.view.display_loaded_workspaces.assert_not_called()


if __name__ == '__main__':
    unittest.main()
