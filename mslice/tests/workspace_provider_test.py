from __future__ import (absolute_import, division, print_function)
import numpy as np
from mock import patch
import unittest
from mslice.models.workspacemanager.workspace_algorithms import (subtract,
                                                                 add_workspace_runs,
                                                                 combine_workspace, rename_workspace,
                                                                 propagate_properties, get_limits, run_alg)
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, get_workspace_names, \
    delete_workspace
from mslice.models.workspacemanager.workspace_algorithms import _processEfixed

class MantidWorkspaceProviderTest(unittest.TestCase):

    def setUp(self):
        self.test_ws_2d = run_alg('CreateSimulationWorkspace', output_name='test_ws_2d', Instrument='MAR',
                                  BinParams=[-10, 1, 10], UnitX='DeltaE')
        run_alg('AddSampleLog', Workspace=self.test_ws_2d.raw_ws, store=False, LogName='Ei', LogText='3.',
                LogType='Number')
        self.test_ws_md = run_alg('ConvertToMD', output_name='test_ws_md', InputWorkspace=self.test_ws_2d,
                                  QDimensions='|Q|', dEAnalysisMode='Direct', MinValues='-10,0,0', MaxValues='10,6,500',
                                  SplitInto='50,50')
        self.test_ws_md.ef_defined = False
        self.test_ws_md.limits = {'DeltaE': [0, 2, 1]}

    def test_delete_workspace(self):
        delete_workspace('test_ws_md')
        self.assertFalse('test_ws_md' in get_workspace_names())

    def test_subtract_workspace(self):
        subtract(['test_ws_2d'], 'test_ws_2d', 0.95)
        result = get_workspace_handle('test_ws_2d_subtracted')
        np.testing.assert_array_almost_equal(result.raw_ws.dataY(0), [0.05] * 20)
        np.testing.assert_array_almost_equal(self.test_ws_2d.raw_ws.dataY(0), [1] * 20)
        self.assertFalse('scaled_bg_ws' in get_workspace_names())

    def test_add_workspace(self):
        add_workspace_runs(['test_ws_2d', 'test_ws_2d'])
        result = get_workspace_handle('test_ws_2d_sum')
        np.testing.assert_array_almost_equal(result.raw_ws.dataY(0), [2.0] * 20)
        np.testing.assert_array_almost_equal(self.test_ws_2d.raw_ws.dataY(0), [1] * 20)

    @patch('mslice.models.workspacemanager.workspace_provider._original_step_size')
    def test_combine_workspace(self, step_mock):
        ws_2 = run_alg('CloneWorkspace', output_name='ws_2', InputWorkspace=self.test_ws_md)
        step_mock.return_value = 1
        combined = combine_workspace([self.test_ws_md, ws_2], 'combined')
        np.testing.assert_array_almost_equal(combined.limits['DeltaE'], [-10, 10, 1], 4)
        np.testing.assert_array_almost_equal(combined.limits['|Q|'], [0.071989, 3.45243, 0.033804], 4)

    def test_process_EFixed(self):
        _processEfixed(self.test_ws_2d)
        self.assertTrue(self.test_ws_2d.ef_defined)

    def test_rename_workspace(self):
        rename_workspace('test_ws_md', 'newname')
        self.assertTrue('newname' in get_workspace_names())
        self.assertFalse('test_ws_md' in get_workspace_names())
        new_ws = get_workspace_handle('newname')
        self.assertFalse(new_ws.ef_defined)
        self.assertEqual(new_ws.limits['DeltaE'], [0, 2, 1])

    def test_propagate_properties(self):
        ws_2 = run_alg('CreateSimulationWorkspace', output_name='test_ws_2', Instrument='MAR', BinParams=[-1, 1, 20],
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

    @patch('mslice.models.workspacemanager.workspace_provider._original_step_size')
    def test_get_limits_100_steps(self, step_mock):
        self.test_ws_md.limits = {}
        step_mock.return_value = 1
        limits = get_limits('test_ws_md', 'DeltaE')
        np.testing.assert_array_almost_equal(limits, [-10, 10, 1], 4)
        limits = get_limits('test_ws_md', '|Q|')
        np.testing.assert_array_almost_equal(limits, [0.06, 3.4645, 0.004], 4)


    def test_get_limits_saved(self):
        self.test_ws_2d.limits = {}
        get_limits('test_ws_2d', 'DeltaE')
        np.testing.assert_array_equal(self.test_ws_2d.limits['DeltaE'], [-10, 10, 1])
        np.testing.assert_array_equal(self.test_ws_2d.limits['|Q|'], self.test_ws_2d.limits['MomentumTransfer'])
        np.testing.assert_almost_equal(self.test_ws_2d.limits['|Q|'], [0.05998867,  3.46446147,  0.00401079], 5)
        np.testing.assert_almost_equal(self.test_ws_2d.limits['Degrees'], [3.43, 134.14, 0.5729578], 5)
