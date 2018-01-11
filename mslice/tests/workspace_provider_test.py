from __future__ import (absolute_import, division, print_function)
import numpy as np
import unittest
from mantid.simpleapi import CreateWorkspace, CreateSimulationWorkspace, AddSampleLog, ConvertToMD
from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider

class MantidWorkspaceProviderTest(unittest.TestCase):

    def setUp(self):
        self.ws_provider = MantidWorkspaceProvider()

        self.test_ws_2d = CreateSimulationWorkspace(Instrument='MAR', BinParams=[-10, 1, 10],
                                                    UnitX='DeltaE', OutputWorkspace='test_ws_2d')
        AddSampleLog(self.test_ws_2d, LogName='Ei', LogText='3.', LogType='Number')
        self.test_ws_md = ConvertToMD(InputWorkspace=self.test_ws_2d, OutputWorkspace="test_ws_md", QDimensions='|Q|',
                                      dEAnalysisMode='Direct', MinValues='-10,0,0', MaxValues='10,6,500',
                                      SplitInto='50,50')

    def test_delete_workspace(self):
        self.ws_provider._EfDefined['test_ws_md'] = False
        self.ws_provider._limits['test_ws_md'] = [0, 2, 1]

        self.ws_provider.delete_workspace('test_ws_md')

        self.assertFalse('test_ws_md' in self.ws_provider.get_workspace_names())
        self.assertRaises(KeyError, lambda: self.ws_provider._EfDefined['test_ws_md'])
        self.assertRaises(KeyError, lambda: self.ws_provider._limits['test_ws_md'])

    def test_process_EFixed(self):
        self.ws_provider._processEfixed('test_ws_2d')
        self.assertTrue(self.ws_provider._EfDefined['test_ws_2d'])

    def test_rename_workspace(self):
        self.ws_provider._EfDefined['test_ws_md'] = False
        self.ws_provider._limits['test_ws_md'] = [0, 2, 1]
        self.ws_provider.rename_workspace('test_ws_md', 'newname')
        self.assertTrue('newname' in self.ws_provider.get_workspace_names())
        self.assertRaises(KeyError, lambda: self.ws_provider._EfDefined['test_ws_md'])
        self.assertRaises(KeyError, lambda: self.ws_provider._limits['test_ws_md'])
        self.assertEqual(False, self.ws_provider._EfDefined['newname'])
        self.assertEqual([0, 2, 1], self.ws_provider._limits['newname'])

    def test_propagate_properties(self):
        x = np.linspace(0, 99, 100)
        y = x * 1
        CreateWorkspace(x, y, OutputWorkspace='test_ws_2')
        self.ws_provider._EfDefined['test_ws_2'] = False
        self.ws_provider._limits['test_ws_2'] = [0, 2, 1]
        self.ws_provider.propagate_properties('test_ws_2', 'test_ws_md')
        self.ws_provider.delete_workspace('test_ws_2')
        self.assertEqual(False, self.ws_provider._EfDefined['test_ws_md'])
        self.assertEqual([0, 2, 1], self.ws_provider._limits['test_ws_md'])

    def test_get_limits(self):
        limits = self.ws_provider.get_limits('test_ws_2d', 'DeltaE')
        self.assertEqual(limits[0], -10)
        self.assertEqual(limits[1], 10)
        self.assertEqual(limits[2], 1)

    def test_get_limits_100_steps(self):
        self.ws_provider._limits['test_ws_md'] = []
        limits = self.ws_provider.get_limits('test_ws_md', 'DeltaE')
        self.assertAlmostEqual(limits[0], -10, places=4)
        self.assertAlmostEqual(limits[1], 10, places=4)
        self.assertAlmostEqual(limits[2], 0.2, places=4)
        limits = self.ws_provider.get_limits('test_ws_md', '|Q|')
        self.assertAlmostEqual(limits[0], 0.07199, places=4)
        self.assertAlmostEqual(limits[1], 3.45243, places=4)
        self.assertAlmostEqual(limits[2], 0.033804, places=4)

    def test_get_limits_saved(self):
        self.ws_provider.get_limits('test_ws_2d', 'DeltaE')
        limits = self.ws_provider._limits['test_ws_2d']
        np.testing.assert_array_equal(limits['DeltaE'], [-10, 10, 1])
        np.testing.assert_array_equal(limits['|Q|'], limits['MomentumTransfer'])
        np.testing.assert_almost_equal(limits['|Q|'], [0.05998867,  3.46446147,  0.01203236], 5)
        np.testing.assert_almost_equal(limits['Degrees'], [3.43, 134.14, 0.5729578], 5)
