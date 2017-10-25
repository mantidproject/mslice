from __future__ import (absolute_import, division, print_function)
import numpy as np
import unittest
from mantid.simpleapi import CreateWorkspace
from mslice.models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider

class MantidWorkspaceProviderTest(unittest.TestCase):

    def setUp(self):
        self.ws_provider = MantidWorkspaceProvider()
        x = np.linspace(0, 99, 100)
        y = x * 1
        CreateWorkspace(x, y, OutputWorkspace='test_ws_provider')


    def test_delete_workspace(self):
        self.ws_provider._EfDefined['test_ws_provider'] = False
        self.ws_provider._limits['test_ws_provider'] = [0, 2, 1]

        self.ws_provider.delete_workspace('test_ws_provider')

        self.assertFalse('test_ws_provider' in self.ws_provider.get_workspace_names())
        self.assertRaises(KeyError, lambda: self.ws_provider._EfDefined['test_ws_provider'])
        self.assertRaises(KeyError, lambda: self.ws_provider._limits['test_ws_provider'])

    def test_process_EFixed(self):
        self.ws_provider._processEfixed('test_ws_provider')
        self.assertFalse(self.ws_provider._EfDefined['test_ws_provider'])


    def test_rename_workspace(self):
        self.ws_provider._EfDefined['test_ws_provider'] = False
        self.ws_provider._limits['test_ws_provider'] = [0, 2, 1]
        self.ws_provider.rename_workspace('test_ws_provider', 'newname')
        self.assertTrue('newname' in self.ws_provider.get_workspace_names())
        self.assertRaises(KeyError, lambda: self.ws_provider._EfDefined['test_ws_provider'])
        self.assertRaises(KeyError, lambda: self.ws_provider._limits['test_ws_provider'])
        self.assertEqual(False, self.ws_provider._EfDefined['newname'])
        self.assertEqual([0, 2, 1], self.ws_provider._limits['newname'])

    def test_propagate_properties(self):
        x = np.linspace(0, 99, 100)
        y = x * 1
        CreateWorkspace(x, y, OutputWorkspace='test_ws_2')
        self.ws_provider._EfDefined['test_ws_2'] = False
        self.ws_provider._limits['test_ws_2'] = [0, 2, 1]
        self.ws_provider.propagate_properties('test_ws_2', 'test_ws_provider')
        self.ws_provider.delete_workspace('test_ws_2')
        self.assertEqual(False, self.ws_provider._EfDefined['test_ws_provider'])
        self.assertEqual([0, 2, 1], self.ws_provider._limits['test_ws_provider'])

    def test_process_load_ws_limits_none(self):
        self.assertEqual(None, self.ws_provider._processLoadedWSLimits('test_ws_provider'))
