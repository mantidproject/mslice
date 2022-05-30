from __future__ import (absolute_import, division, print_function)
import unittest

import mock
from mock import call, patch
import numpy as np

from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.presenters.workspace_manager_presenter import WorkspaceManagerPresenter
from mslice.views.interfaces.mainview import MainView
from mslice.views.interfaces.workspace_view import WorkspaceView
from mslice.widgets.workspacemanager.command import Command
from mslice.widgets.workspacemanager import TAB_2D, TAB_NONPSD
from mslice.util.mantid.mantid_algorithms import AddSampleLog, CreateWorkspace, CreateSimulationWorkspace,\
    ConvertToMD, CloneWorkspace


class WorkspaceManagerPresenterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x = np.linspace(0, 99, 100)
        y = x * 1
        e = y * 0 + 2
        cls.m_workspace = CreateWorkspace(x, y, e, OutputWorkspace="m_ws")

        cls.sim_workspace = CreateSimulationWorkspace(Instrument='MAR', BinParams=[-10, 1, 10],
                                                      UnitX='DeltaE', OutputWorkspace='ws0')
        AddSampleLog(cls.sim_workspace, LogName='Ei', LogText='3.', LogType='Number')
        cls.px_workspace = ConvertToMD(InputWorkspace=cls.sim_workspace, OutputWorkspace="ws1", QDimensions='|Q|',
                                       dEAnalysisMode='Direct', MinValues='-10,0,0', MaxValues='10,6,500',
                                       SplitInto='50,50')
        cls.px_workspace_clone = CloneWorkspace(InputWorkspace=cls.px_workspace, OutputWorkspace='ws2',
                                                StoreInADS=False)

    def setUp(self):
        self.view = mock.create_autospec(spec=WorkspaceView)
        self.mainview = mock.create_autospec(MainView)
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.mainview.get_presenter = mock.Mock(return_value=self.main_presenter)

    def test_register_master_success(self):
        workspace_presenter = WorkspaceManagerPresenter(self.view)
        workspace_presenter.register_master(self.main_presenter)
        self.main_presenter.register_workspace_selector.assert_called_once_with(workspace_presenter)

    def test_register_master_invalid_master_fail(self):
        workspace_presenter = WorkspaceManagerPresenter(self.view)
        self.assertRaises(AssertionError, workspace_presenter.register_master, 3)

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    @patch('mslice.presenters.workspace_manager_presenter.save_workspaces')
    def test_save_workspace(self, save_ws_mock, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = r'A:\file\path'
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        save_dir_mock.return_value = (path_to_save_to, workspace_to_save, '.nxs')

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False,
                                              default_ext='.nxs')
        save_ws_mock.assert_called_once_with([workspace_to_save], path_to_save_to, 'file1', '.nxs')

        # Test save failure
        save_ws_mock.side_effect = RuntimeError()
        self.view.error_unable_to_save = mock.Mock()
        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.error_unable_to_save.assert_called_once_with()

        # Test user cancellation
        save_dir_mock.reset_mock()
        save_ws_mock.reset_mock()
        save_dir_mock.side_effect = RuntimeError('dialog cancelled')
        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        save_ws_mock.assert_not_called()

        # Test other dialog failure
        save_dir_mock.side_effect = RuntimeError()
        with self.assertRaises(RuntimeError):
            self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        save_ws_mock.assert_not_called()

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    @patch('mslice.presenters.workspace_manager_presenter.save_workspaces')
    def test_save_ascii_workspace(self, save_ws_mock, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = r'A:\file\path'
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        save_dir_mock.return_value = (path_to_save_to, workspace_to_save, '.txt')
        self.presenter.notify(Command.SaveSelectedWorkspaceAscii)
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False, default_ext='.txt')
        self.view.get_workspace_selected.assert_called_once_with()
        save_ws_mock.assert_called_once_with([workspace_to_save], path_to_save_to, 'file1', '.txt')

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    @patch('mslice.presenters.workspace_manager_presenter.save_workspaces')
    def test_save_matlab_workspace(self, save_ws_mock, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = r'A:\file\path'
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        save_dir_mock.return_value = (path_to_save_to, workspace_to_save, '.mat')

        self.presenter.notify(Command.SaveSelectedWorkspaceMatlab)
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False, default_ext='.mat')
        save_ws_mock.assert_called_once_with([workspace_to_save], path_to_save_to, 'file1', '.mat')

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    @patch('mslice.presenters.workspace_manager_presenter.save_workspaces')
    def test_save_workspace_multiple_selected(self, save_ws_mock, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that reports multiple workspaces are selected on calls to get_workspace_selected
        path_to_save_to = r'A:\file\path'
        self.view.get_workspace_selected = mock.Mock(return_value=['file1', 'file2'])
        save_dir_mock.return_value = (path_to_save_to, None, '.nxs')
        save_ws_mock.side_effect = [True, RuntimeError]

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=True, save_as_image=False, default_ext='.nxs')
        save_ws_mock.assert_called_with(['file1', 'file2'], path_to_save_to, None, '.nxs')

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    @patch('mslice.presenters.workspace_manager_presenter.save_workspaces')
    def test_save_workspace_non_selected_prompt_user(self, save_ws_mock, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that reports no workspaces arw selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        save_dir_mock.assert_not_called()
        save_ws_mock.assert_not_called()

    @patch('mslice.presenters.workspace_manager_presenter.get_save_directory')
    @patch('mslice.presenters.workspace_manager_presenter.save_workspaces')
    def test_save_workspace_cancelled(self, save_ws_mock, save_dir_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that report a single selected workspace on calls to get_workspace_selected and supplies a path
        # to save to on calls to get_workspace_to_save_filepath
        path_to_save_to = ""  # view returns empty string to indicate operation cancelled
        workspace_to_save = 'file1'
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_save])
        self.view.error_invalid_save_path = mock.Mock()
        save_dir_mock.return_value = (path_to_save_to, workspace_to_save, '.nxs')

        self.presenter.notify(Command.SaveSelectedWorkspaceNexus)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_selected.assert_called_once_with()
        save_dir_mock.assert_called_once_with(multiple_files=False, save_as_image=False, default_ext='.nxs')
        self.view.error_invalid_save_path.assert_called_once()
        save_ws_mock.assert_not_called()

    @patch('mslice.presenters.workspace_manager_presenter.delete_workspace')
    def test_remove_workspace(self, delete_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a workspace that reports a single selected workspace on calls to get_workspace_selected
        workspace_to_be_removed = CloneWorkspace(self.m_workspace.raw_ws, OutputWorkspace='file1')
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace_to_be_removed])
        self.view.display_loaded_workspaces = mock.Mock()

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        delete_ws_mock.assert_called_once_with(workspace_to_be_removed)
        self.view.display_loaded_workspaces.assert_called_once()

    @patch('mslice.presenters.workspace_manager_presenter.delete_workspace')
    def test_remove_multiple_workspaces(self, delete_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that reports 3 selected workspaces on calls to get_workspace_selected
        workspace1 = CloneWorkspace(self.m_workspace.raw_ws, OutputWorkspace='file1')
        workspace2 = CloneWorkspace(self.m_workspace.raw_ws, OutputWorkspace='file2')
        self.view.get_workspace_selected = mock.Mock(return_value=[workspace1, workspace2])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        delete_calls = [call(workspace1), call(workspace2)]
        delete_ws_mock.assert_has_calls(delete_calls, any_order=True)
        assert(self.view.display_loaded_workspaces.called)

    @patch('mslice.presenters.workspace_manager_presenter.delete_workspace')
    def test_remove_all_workspaces(self, delete_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that reports 3 selected workspaces on calls to get_workspace_selected
        workspace1 = CloneWorkspace(self.m_workspace.raw_ws, OutputWorkspace='file1')
        workspace2 = CloneWorkspace(self.m_workspace.raw_ws, OutputWorkspace='file2')
        #self.view.get_workspace_selected = mock.Mock(return_value=[workspace1, workspace2])

        self.presenter.notify(Command.RemoveAllWorkspaces)
        delete_calls = [call(workspace1), call(workspace2)]
        delete_ws_mock.assert_has_calls(delete_calls, any_order=True)
        assert(self.view.display_loaded_workspaces.called)

    @patch('mslice.presenters.workspace_manager_presenter.delete_workspace')
    def test_remove_workspace_non_selected_prompt_user(self, delete_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that reports no workspace selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.RemoveSelectedWorkspaces)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()
        delete_ws_mock.assert_not_called()
        assert(not self.view.display_loaded_workspaces.called)

    def test_broadcast_success(self):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.presenter.register_master(self.main_presenter)
        self.presenter.notify(Command.SelectionChanged)
        self.main_presenter.notify_workspace_selection_changed()

    def test_workspace2d_nonpsd_selection(self):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.presenter.register_master(self.main_presenter)
        self.view.get_workspace_selected = mock.Mock(return_value=[self.m_workspace])
        self.view.current_tab = mock.Mock(return_value=TAB_2D)
        self.view.tab_changed = mock.Mock()
        self.presenter._psd = True
        self.presenter.notify(Command.SelectionChanged)
        self.main_presenter.notify_workspace_selection_changed()
        self.view.tab_changed.emit.assert_called_once_with(TAB_NONPSD)

    def test_workspace2d_psd_selection(self):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.presenter.register_master(self.main_presenter)
        self.view.current_tab = mock.Mock(return_value=TAB_2D)
        self.view.get_workspace_selected = mock.Mock(return_value=[self.sim_workspace])
        self.sim_workspace.is_PSD = mock.Mock(return_value=True)
        self.view.tab_changed = mock.Mock()
        self.presenter._psd = False
        self.presenter.notify(Command.SelectionChanged)
        self.main_presenter.notify_workspace_selection_changed()
        self.view.tab_changed.emit.assert_called_once_with(TAB_2D)

    def test_call_presenter_with_unknown_command(self):
        self.presenter = WorkspaceManagerPresenter(self.view)
        unknown_command = 10
        self.assertRaises(ValueError, self.presenter.notify, unknown_command)

    def test_notify_presenter_clears_error(self):
        presenter = WorkspaceManagerPresenter(self.view)
        presenter.register_master(self.main_presenter)
        # This unit test will verify that notifying cut presenter will cause the error to be cleared on the view.
        # The actual subsequent procedure will fail, however this irrelevant to this. Hence the try, except blocks
        for command in [x for x in dir(Command) if x[0] != "_"]:
            try:
                presenter.notify(command)
            except ValueError:
                pass
            assert(self.view.clear_displayed_error.called)
            self.view.reset_mock()

    def test_set_selected_workspace_index(self):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.view.get_workspace_index = mock.Mock()
        self.presenter.set_selected_workspaces([1])
        self.view.set_workspace_selected.assert_called_once_with([1])

    def test_set_selected_workspace_name(self):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.view.get_workspace_index = mock.Mock(return_value=0)
        self.presenter.set_selected_workspaces(['ws'])
        self.view.get_workspace_index.assert_called_once_with('ws')
        self.view.set_workspace_selected.assert_called_once_with([0])

    @patch('mslice.presenters.workspace_manager_presenter.get_workspace_name')
    def test_set_selected_workspace_handle(self, get_ws_name_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.view.get_workspace_index = mock.Mock(return_value=0)
        get_ws_name_mock.return_value = 'ws'
        self.presenter.set_selected_workspaces([mock.Mock()])
        get_ws_name_mock.called_once_with(mock.Mock())
        self.view.get_workspace_index.assert_called_once_with('ws')
        self.view.set_workspace_selected.assert_called_once_with([0])

    @patch('mslice.presenters.workspace_manager_presenter.combine_workspace')
    @patch('mslice.presenters.workspace_manager_presenter.is_pixel_workspace')
    def test_combine_workspace_no_ws(self, is_pixel_ws_mock, combine_ws_mock):
        # Checks that it will fail if only no workspaces are selected.
        is_pixel_ws_mock.return_value = True
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = []
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.presenter.notify(Command.CombineWorkspace)
        self.view.error_select_more_than_one_workspaces.assert_called_once_with()

    @patch('mslice.presenters.workspace_manager_presenter.combine_workspace')
    @patch('mslice.presenters.workspace_manager_presenter.is_pixel_workspace')
    def test_combine_workspace_single_ws(self, is_pixel_ws_mock, combine_ws_mock):
        # Checks that it will fail if only one workspace is selected.
        is_pixel_ws_mock.return_value = True
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = ['ws1']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.view.add_workspace_dialog = mock.Mock(return_value='ws2')
        self.presenter.notify(Command.CombineWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.add_workspace_dialog.assert_called_once()
        combine_ws_mock.assert_called_once_with(['ws1', 'ws2'], 'ws1_combined')

    @patch('mslice.presenters.workspace_manager_presenter.is_pixel_workspace')
    def test_combine_workspace_wrong_type(self, is_pixel_ws_mock):
        # Checks that it will fail if one of the workspace is not a MDEventWorkspace
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        is_pixel_ws_mock.side_effect = [True, False]
        self.presenter.notify(Command.CombineWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        check_calls = [call('ws1'), call('ws2')]
        is_pixel_ws_mock.assert_has_calls(check_calls, any_order=True)
        assert(self.view.error_select_more_than_one_workspaces.called)

    @patch('mslice.presenters.workspace_manager_presenter.combine_workspace')
    @patch('mslice.presenters.workspace_manager_presenter.is_pixel_workspace')
    def test_combine_workspace(self, is_pixel_ws_mock, combine_ws_mock):
        # Now checks it succeeds otherwise
        is_pixel_ws_mock.return_value = True
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)

        self.presenter.notify(Command.CombineWorkspace)
        self.view.get_workspace_selected.assert_called()
        assert(not self.view.error_select_more_than_one_workspaces.called)
        combine_ws_mock.assert_called_once_with(selected_workspaces, selected_workspaces[0]+'_combined')

    @patch('mslice.presenters.workspace_manager_presenter.add_workspace_runs')
    def test_add_workspaces(self, add_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.view.get_workspace_selected = mock.Mock(return_value=[])
        self.presenter.notify(Command.Add)
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()

        self.view.get_workspace_selected.return_value = ['ws1']
        self.view.add_workspace_dialog.return_value = 'ws2'
        self.view._display_error = mock.Mock()
        add_ws_mock.side_effect = ValueError('incompatible workspace')
        self.presenter.notify(Command.Add)
        add_ws_mock.assert_called_once_with(['ws1', 'ws2'])
        self.view._display_error.assert_called_once_with('incompatible workspace')

    @patch('mslice.presenters.workspace_manager_presenter.subtract')
    def test_subtract_workspaces(self, subtract_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.view.get_workspace_selected = mock.Mock(return_value=[])
        self.presenter.notify(Command.Subtract)
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()

        self.view.get_workspace_selected.return_value = ['ws1']
        self.view.subtraction_input = mock.Mock(side_effect=RuntimeError)
        self.presenter.notify(Command.Subtract)
        subtract_ws_mock.assert_not_called()

        self.view.subtraction_input.side_effect = None
        self.view.subtraction_input.return_value = ('ws2', 0.5)
        self.view._display_error = mock.Mock()
        subtract_ws_mock.side_effect = ValueError('incompatible workspace')
        self.presenter.notify(Command.Subtract)
        subtract_ws_mock.assert_called_once_with(['ws1'], 'ws2', 0.5)
        self.view._display_error.assert_called_once_with('incompatible workspace')

    @patch('mslice.presenters.workspace_manager_presenter.export_workspace_to_ads')
    def test_save_to_ads(self, export_to_ads_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.presenter.notify(Command.SaveToADS)
        export_to_ads_mock.assert_has_calls([mock.call(ws) for ws in selected_workspaces])

        self.view.get_workspace_selected.return_value = []
        self.presenter.notify(Command.SaveToADS)
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()

    @patch('mslice.presenters.workspace_manager_presenter.scale_workspaces')
    def test_scale_workspace(self, scale_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.view.scale_input = mock.Mock(return_value=(0.5, None))

        self.presenter.notify(Command.Scale)
        self.view.get_workspace_selected.assert_called()
        self.view.scale_input.assert_called()
        assert(not self.view.error_select_more_than_one_workspaces.called)
        scale_ws_mock.assert_called_once_with(selected_workspaces, scale_factor=0.5)

    @patch('mslice.presenters.workspace_manager_presenter.scale_workspaces')
    def test_rebose_workspace(self, scale_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.view.scale_input = mock.Mock(return_value=(250, 2))

        self.presenter.notify(Command.Bose)
        self.view.get_workspace_selected.assert_called()
        self.view.scale_input.assert_called()
        assert(not self.view.error_select_more_than_one_workspaces.called)
        scale_ws_mock.assert_called_once_with(selected_workspaces, from_temp=250, to_temp=2)

    @patch('mslice.presenters.workspace_manager_presenter.scale_workspaces')
    def test_scale_workspace_errors(self, scale_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        selected_workspaces = ['ws1', 'ws2']

        self.view.get_workspace_selected = mock.Mock(return_value=[])
        self.presenter.notify(Command.Scale)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_or_more_workspaces.assert_called_once_with()

        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.view.scale_input = mock.Mock(side_effect=RuntimeError())
        self.presenter.notify(Command.Scale)
        self.view.get_workspace_selected.assert_called()
        self.view.scale_input.assert_called()
        assert(not self.view.error_select_more_than_one_workspaces.called)
        assert(not scale_ws_mock.called)

        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)
        self.view.scale_input = mock.Mock(return_value=(0.5, None))
        self.view._display_error = mock.Mock()
        scale_ws_mock.side_effect = ValueError()
        self.presenter.notify(Command.Scale)
        self.view.get_workspace_selected.assert_called()
        self.view.scale_input.assert_called()
        assert(not self.view.error_select_more_than_one_workspaces.called)
        scale_ws_mock.assert_called_once_with(selected_workspaces, scale_factor=0.5)
        self.view._display_error.assert_called()

    def test_compose_error(self):
        self.presenter = WorkspaceManagerPresenter(self.view)
        self.view._display_error = mock.Mock()
        self.presenter.notify(Command.ComposeWorkspace)
        self.view._display_error.assert_called()


if __name__ == '__main__':
    unittest.main()
