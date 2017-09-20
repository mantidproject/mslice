import numpy as np
import unittest

from mantid.simpleapi import CreateWorkspace, CreateMDHistoWorkspace

from mslice.workspace.histogram_workspace import HistogramWorkspace
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


class WorkspaceTest(BaseWorkspaceTest):

    @classmethod
    def setUpClass(cls):
        x = np.linspace(0, 99, 100)
        y = x * 1
        e = y * 0 + 2
        cls.workspace = Workspace(CreateWorkspace(x, y, e, OutputWorkspace="testBaseWorkspace"))

    def test_get_coordinates(self):
        expected_values = np.linspace(0, 99, 100)
        self.assertTrue((expected_values == self.workspace.get_coordinates()['']).all())

    def test_get_signal(self):
        expected_values = range(0, 100)
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


class HistogramWorkspaceTest(BaseWorkspaceTest):

    @classmethod
    def setUpClass(cls):
        signal = range(0, 100)
        error = np.zeros(100) + 2
        cls.workspace = HistogramWorkspace(CreateMDHistoWorkspace(Dimensionality=2, Extents='0,100,0,100',
                                                                  SignalInput=signal, ErrorInput=error,
                                                                  NumberOfBins='10,10', Names='Dim1,Dim2',
                                                                  Units='U,U', OutputWorkspace='testHistoWorkspace'))

    def test_get_coordinates(self):
        expected = np.linspace(0, 100, 10)
        self.assertTrue((self.workspace.get_coordinates()['Dim1'] == expected).all())

    def test_get_signal(self):
        self.check_signal()

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
        list_to_add = np.linspace(0, -9, 10)
        result = self.workspace + list_to_add
        result = result.get_signal()

        line = np.multiply(np.linspace(0, 9, 10), 10)
        expected_values = np.empty((10, 10))
        for i in range(10):
            expected_values[i] = line

        self.assertTrue((result == expected_values).all())

    def test_add_invalid_list(self):
        invalid_list = np.linspace(0, -6, 3)
        self.assertRaises(RuntimeError, lambda: self.workspace + invalid_list)
