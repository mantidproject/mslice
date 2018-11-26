from __future__ import (absolute_import, division, print_function)
import numpy as np
import mock
from mock import patch
import unittest
from mslice.models.workspacemanager.workspace_algorithms import (subtract, add_workspace_runs, combine_workspace,
                                                                 propagate_properties, get_limits)
from mslice.models.workspacemanager.workspace_provider import (get_workspace_handle, get_visible_workspace_names,
                                                               delete_workspace, rename_workspace)
from mslice.models.workspacemanager.workspace_algorithms import processEfixed
from mslice.util.mantid.mantid_algorithms import ConvertToMD, CloneWorkspace, CreateSimulationWorkspace
from mslice.widgets.workspacemanager.command import Command
from mantid.simpleapi import AddSampleLog
from mslice.presenters.workspace_manager_presenter import WorkspaceManagerPresenter


class MantidWorkspaceProviderTest(unittest.TestCase):

    def setUp(self):
        self.test_ws_2d = CreateSimulationWorkspace(OutputWorkspace='test_ws_2d', Instrument='MAR',
                                                    BinParams=[-10, 1, 10], UnitX='DeltaE')
        AddSampleLog(Workspace=self.test_ws_2d.raw_ws, LogName='Ei', LogText='50.',
                     LogType='Number', StoreInADS=False)
        self.test_ws_md = ConvertToMD(OutputWorkspace='test_ws_md', InputWorkspace=self.test_ws_2d,
                                      QDimensions='|Q|', dEAnalysisMode='Direct', MinValues='-10,0,0', MaxValues='10,6,500',
                                      SplitInto='50,50')
        self.test_ws_2d.e_mode = "Direct"
        self.test_ws_md.ef_defined = False
        self.test_ws_md.is_PSD = True
        self.test_ws_md.e_mode = "Direct"
        self.test_ws_md.limits = {'DeltaE': [0, 2, 1]}

    def test_delete_workspace(self):
        delete_workspace('test_ws_md')
        self.assertFalse('test_ws_md' in get_visible_workspace_names())

    def test_subtract_workspace(self):
        subtract(['test_ws_2d'], 'test_ws_2d', 0.95)
        result = get_workspace_handle('test_ws_2d_subtracted')
        np.testing.assert_array_almost_equal(result.raw_ws.dataY(0), [0.05] * 20)
        np.testing.assert_array_almost_equal(self.test_ws_2d.raw_ws.dataY(0), [1] * 20)
        self.assertFalse('scaled_bg_ws' in get_visible_workspace_names())

    def test_add_workspace(self):
        original_data = self.test_ws_2d.raw_ws.dataY(0)
        add_workspace_runs(['test_ws_2d', 'test_ws_2d'])
        result = get_workspace_handle('test_ws_2d_sum')
        np.testing.assert_array_almost_equal(result.raw_ws.dataY(0), [2.0] * 20)
        np.testing.assert_array_almost_equal(original_data, [1] * 20)

    @patch('mslice.models.workspacemanager.workspace_algorithms._original_step_size')
    def test_combine_workspace(self, step_mock):
        ws_2 = CloneWorkspace(OutputWorkspace='ws_2', InputWorkspace=self.test_ws_md)
        ws_2.e_mode = "Direct"
        step_mock.return_value = 1
        combined = combine_workspace([self.test_ws_md, ws_2], 'combined')
        np.testing.assert_array_almost_equal(combined.limits['DeltaE'], [-10, 10, 1], 4)
        np.testing.assert_array_almost_equal(combined.limits['|Q|'], [0.2939, 9.4817, 0.0919], 4)
        self.assertTrue(combined.is_PSD)

    def test_process_EFixed(self):
        processEfixed(self.test_ws_2d)
        self.assertTrue(self.test_ws_2d.ef_defined)

    def test_rename_workspace(self):
        rename_workspace('test_ws_md', 'newname')
        self.assertTrue('newname' in get_visible_workspace_names())
        self.assertFalse('test_ws_md' in get_visible_workspace_names())
        new_ws = get_workspace_handle('newname')
        self.assertFalse(new_ws.ef_defined)
        self.assertEqual(new_ws.limits['DeltaE'], [0, 2, 1])

    @patch('mslice.models.workspacemanager.workspace_provider.rename_workspace')
    @patch('mslice.models.workspacemanager.workspace_provider.get_visible_workspace_names')
    def test_rename_workspace(self, get_ws_names_mock, rename_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that will return a single selected workspace on call to get_workspace_selected and supply a
        # name on call to get_workspace_new_name
        old_workspace_name = 'file1'
        new_workspace_name = 'new_name'
        self.view.get_workspace_selected = mock.Mock(return_value=[old_workspace_name])
        self.view.get_workspace_new_name = mock.Mock(return_value=new_workspace_name)
        self.view.display_loaded_workspaces = mock.Mock()
        get_ws_names_mock.return_value = ['file1', 'file2', 'file3']

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.get_workspace_new_name.assert_called_once_with()
        rename_ws_mock.assert_called_once_with('file1', 'new_name')
        self.view.display_loaded_workspaces.assert_called_once()

    @patch('mslice.models.workspacemanager.workspace_provider.rename_workspace')
    def test_rename_workspace_multiple_workspace_selected_prompt_user(self, rename_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that reports multiple selected workspaces on calls to get_workspace_selected
        selected_workspaces = ['ws1', 'ws2']
        self.view.get_workspace_selected = mock.Mock(return_value=selected_workspaces)

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_only_one_workspace.assert_called_once_with()
        rename_ws_mock.assert_not_called()

    @patch('mslice.models.workspacemanager.workspace_provider.rename_workspace')
    def test_rename_workspace_non_selected_prompt_user(self, rename_ws_mock):
        self.presenter = WorkspaceManagerPresenter(self.view)
        # Create a view that reports that no workspaces are selected on calls to get_workspace_selected
        self.view.get_workspace_selected = mock.Mock(return_value=[])

        self.presenter.notify(Command.RenameWorkspace)
        self.view.get_workspace_selected.assert_called_once_with()
        self.view.error_select_one_workspace.assert_called_once_with()
        rename_ws_mock.assert_not_called()

    def test_propagate_properties(self):
        ws_2 = CreateSimulationWorkspace(OutputWorkspace='test_ws_2', Instrument='MAR', BinParams=[-1, 1, 20],
                                         UnitX='DeltaE')
        propagate_properties(self.test_ws_md, ws_2)
        delete_workspace('test_ws_md')
        self.assertFalse(ws_2.ef_defined)
        self.assertEqual({'DeltaE': [0, 2, 1]}, ws_2.limits)

    def test_get_limits(self):
        limits = get_limits('test_ws_md', 'DeltaE')
        self.assertEqual(limits[0], 0)
        self.assertEqual(limits[1], 2)
        self.assertEqual(limits[2], 1)

    @patch('mslice.models.workspacemanager.workspace_algorithms._original_step_size')
    def test_get_limits_100_steps(self, step_mock):
        self.test_ws_md.limits = {}
        step_mock.return_value = 1
        limits = get_limits('test_ws_md', 'DeltaE')
        np.testing.assert_array_almost_equal(limits, [-10, 10, 1], 4)
        limits = get_limits('test_ws_md', '|Q|')
        np.testing.assert_allclose(limits, [0.545576, 8.615743, 0.042867], rtol=0, atol=1e-3)

    def test_get_limits_saved(self):
        self.test_ws_2d.limits = {}
        get_limits('test_ws_2d', 'DeltaE')
        np.testing.assert_array_equal(self.test_ws_2d.limits['DeltaE'], [-10, 10, 1])
        np.testing.assert_array_equal(self.test_ws_2d.limits['|Q|'], self.test_ws_2d.limits['MomentumTransfer'])
        np.testing.assert_almost_equal(self.test_ws_2d.limits['|Q|'], [0.25116,  9.52454,  0.04287], 5)
        np.testing.assert_almost_equal(self.test_ws_2d.limits['2Theta'], [3.43, 134.14, 0.57296], 5)
