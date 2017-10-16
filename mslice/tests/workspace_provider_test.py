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
        e = y * 0 + 2
        CreateWorkspace(x, y, e, OutputWorkspace='test_ws_provider')


    def test_delete_workspace(self):
        self.ws_provider._EfDefined['test_ws_provider'] = False
        self.ws_provider._limits['test_ws_provider'] = [0, 2, 1]

        self.ws_provider.delete_workspace('test_ws_provider')

        self.assertEqual([], self.ws_provider.get_workspace_names())
        self.assertRaises(KeyError, lambda: self.ws_provider._EfDefined['test_ws_provider'])
        self.assertRaises(KeyError, lambda: self.ws_provider._limits['test_ws_provider'])

    def test_process_EFixed(self):
        pass