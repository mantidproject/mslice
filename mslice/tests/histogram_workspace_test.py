from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.simpleapi import CreateMDHistoWorkspace

from mslice.tests.workspace_test import BaseWorkspaceTest
from mslice.workspace.histogram_workspace import HistogramWorkspace


class HistogramWorkspaceTest(BaseWorkspaceTest):

    @classmethod
    def setUpClass(cls):
        signal = list(range(0, 100))
        error = np.zeros(100) + 2
        cls.workspace = HistogramWorkspace(CreateMDHistoWorkspace(Dimensionality=2, Extents='0,100,0,100',
                                                                  SignalInput=signal, ErrorInput=error,
                                                                  NumberOfBins='10,10', Names='Dim1,Dim2',
                                                                  Units='U,U', OutputWorkspace='testHistoWorkspace'))

    def test_invalid_workspace(self):
        self.assertRaises(TypeError, lambda: HistogramWorkspace(4))

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
        np.testing.assert_array_almost_equal(expected_values, result, 8)

    def test_add_invalid_list(self):
        invalid_list = np.linspace(0, -6, 3)
        self.assertRaises(RuntimeError, lambda: self.workspace + invalid_list)
