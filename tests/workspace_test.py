from __future__ import (absolute_import, division, print_function)
import unittest
import mock

import numpy as np
from mantid.simpleapi import CreateWorkspace

from mslice.workspace.helperfunctions import attribute_to_log
from mslice.workspace.workspace import Workspace


class BaseWorkspaceTest(unittest.TestCase):

    def check_signal(self):
        expected_values = np.arange(0, 100)
        result = np.array(self.workspace.get_signal().flatten())
        result.sort()
        self.assertTrue((result == expected_values).all())

    def check_error(self):
        expected = np.zeros(100) + 2
        self.assertTrue((expected == self.workspace.get_error().flatten()).all())

    def check_variance(self):
        expected = np.zeros(100) + 4
        self.assertTrue((expected == self.workspace.get_variance().flatten()).all())

    def check_add_workspace(self):
        two_workspace = self.workspace + self.workspace
        expected_values = np.linspace(0, 198, 100)
        result = np.array(two_workspace.get_signal().flatten())
        result.sort()
        self.assertTrue((result == expected_values).all())

    def check_mul_workspace(self):
        two_workspace = self.workspace * 2
        expected_values = np.linspace(0, 198, 100)
        result = np.array(two_workspace.get_signal().flatten())
        result.sort()
        self.assertTrue((result == expected_values).all())

    def check_pow_workspace(self):
        squared_workspace = self.workspace ** 2
        expected_values = np.square(np.linspace(0, 99, 100))
        result = np.array(squared_workspace.get_signal().flatten())
        result.sort()
        self.assertTrue((result == expected_values).all())

    def check_neg_workspace(self):
        negative_workspace = -self.workspace
        expected_values = np.linspace(-99, 0, 100)
        result = np.array(negative_workspace.get_signal().flatten())
        result.sort()
        self.assertTrue((result == expected_values).all())

    def set_attribute(self):
        self.attr = {'axes': [1, object]}
        attribute_to_log(self.attr, self.workspace.raw_ws)

    def check_attribute_propagation(self, new_workspace):
        assert (hasattr(new_workspace, 'axes'))
        assert (new_workspace.axes == self.attr['axes'])


class WorkspaceTest(BaseWorkspaceTest):

    @classmethod
    def setUpClass(cls):
        x = np.linspace(0, 99, 100)
        y = x * 1
        e = y * 0 + 2
        cls.workspace = Workspace(CreateWorkspace(x, y, e, OutputWorkspace="testBaseWorkspace"), 'testBaseWorkspace')

    def test_invalid_workspace(self):
        self.assertRaisesRegex(TypeError, "Workspace expected matrixWorkspace, got int", lambda: Workspace(4, "WorkspaceName"))

    def test_get_coordinates(self):
        expected_values = np.linspace(0, 99, 100)
        self.assertTrue((expected_values == self.workspace.get_coordinates()['']).all())

    def test_get_signal(self):
        expected_values = list(range(0, 100))
        result = np.array(self.workspace.get_signal().flatten())
        self.assertTrue((result == expected_values).all())

    def test_get_error(self):
        self.check_error()

    def test_get_variance(self):
        self.check_variance()

    def test_add_workspace(self):
        self.check_add_workspace()

    def test_mul_workspace_number(self):
        self.check_mul_workspace()

    def test_pow_workspace(self):
        self.check_pow_workspace()

    def test_neg_workspace(self):
        self.check_neg_workspace()

    def test_add_list(self):
        list_to_add = np.linspace(0, 99, 100)
        result = self.workspace + list_to_add
        result = result.get_signal()
        expected_values = np.multiply(list_to_add, 2)
        self.assertTrue((result == expected_values).all())

    def test_attribute_propagation(self):
        self.set_attribute()
        new_workspace = Workspace(self.workspace.raw_ws, 'new')
        self.check_attribute_propagation(new_workspace)

    def test_loader_name_returns_none_when_the_load_algorithm_is_not_in_the_algorithm_history(self):
        mock_history = mock.MagicMock()
        mock_history.empty.return_value = True

        self.workspace._raw_ws.getHistory = mock.MagicMock(return_value=mock_history)

        self.assertEqual(None, self.workspace.loader_name())

    def test_loader_name_returns_LoadNXSPE_when_it_is_found_in_the_first_alg_history(self):
        load_history = mock.MagicMock()
        load_history.name.return_value = "Load"
        load_history.getPropertyValue.return_value = "LoadNXSPE"

        mock_history = mock.MagicMock()
        mock_history.empty.return_value = False
        mock_history.getAlgorithmHistories.return_value = [load_history]

        self.workspace._raw_ws.getHistory = mock.MagicMock(return_value=mock_history)

        self.assertEqual("LoadNXSPE", self.workspace.loader_name())

    def test_loader_name_returns_none_when_it_load_is_not_the_first_alg_in_the_history(self):
        rebin_history = mock.MagicMock()
        rebin_history.name.return_value = "Rebin"
        load_history = mock.MagicMock()
        load_history.name.return_value = "Load"
        load_history.getPropertyValue.return_value = "LoadNXSPE"

        mock_history = mock.MagicMock()
        mock_history.empty.return_value = False
        mock_history.getAlgorithmHistories.return_value = [rebin_history, load_history]

        self.workspace._raw_ws.getHistory = mock.MagicMock(return_value=mock_history)

        self.assertEqual(None, self.workspace.loader_name())
