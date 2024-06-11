from __future__ import (absolute_import, division, print_function)
import numpy as np

from mantid.simpleapi import CreateMDHistoWorkspace

from tests.workspace_test import BaseWorkspaceTest
from mslice.models.workspacemanager.workspace_provider import add_workspace, remove_workspace
from mslice.workspace.histogram_workspace import HistogramWorkspace


class HistogramWorkspaceTest(BaseWorkspaceTest):

    @classmethod
    def setUpClass(cls):
        signal = list(range(0, 100))
        error = np.zeros(100) + 2
        cls.workspace = HistogramWorkspace(CreateMDHistoWorkspace(Dimensionality=2, Extents='0,100,0,100',
                                                                  SignalInput=signal, ErrorInput=error,
                                                                  NumberOfBins='10,10', Names='Dim1,Dim2',
                                                                  Units='U,U', OutputWorkspace='testHistoWorkspace',
                                                                  ), 'testHistoWorkspace')
        cls.workspace1Bin = HistogramWorkspace(CreateMDHistoWorkspace(Dimensionality=2, Extents='0,100,0,100',
                                                                      SignalInput=signal, ErrorInput=error,
                                                                      NumberOfBins='1,100', Names='Dim1,Dim2',
                                                                      Units='U,U', OutputWorkspace='testHistoWorkspace1Bin',
                                                                      ), 'testHistoWorkspace1Bin')

    def test_invalid_workspace(self):
        self.assertRaisesRegex(TypeError, "HistogramWorkspace expected IMDHistoWorkspace, got int", lambda: HistogramWorkspace(4, 'name'))

    def test_convert_to_matrix(self):
        # workspace needs to be registered with mslice for conversion
        try:
            add_workspace(self.workspace, self.workspace.name)
            matrix_ws = self.workspace.convert_to_matrix()

            self.assertEqual(10, matrix_ws.raw_ws.getNumberHistograms())
            self.assertEqual(10, matrix_ws.raw_ws.blocksize())
        finally:
            # remove mslice tracking
            remove_workspace(self.workspace)

    def test_convert_to_matrix_error(self):
        # workspace needs to be registered with mslice for conversion
        try:
            add_workspace(self.workspace1Bin, self.workspace1Bin.name)
            with self.assertRaises(TypeError):
                self.workspace1Bin.convert_to_matrix()
        finally:
            # remove mslice tracking
            remove_workspace(self.workspace1Bin)

    def test_rename_workspace_which_contains_special_character(self):
        self.workspace.name = "specialcharacter)"
        self.workspace.name = "secondname"

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

        ws_representative_array = np.linspace(0, 99, 100).reshape(10, 10).T
        expected_result = ws_representative_array + list_to_add

        np.testing.assert_array_almost_equal(expected_result, result, 8)
        np.testing.assert_array_almost_equal(ws_representative_array, self.workspace.get_signal(), 8)

    def test_add_invalid_list(self):
        invalid_list = np.linspace(0, -6, 3)
        self.assertRaises(RuntimeError, lambda: self.workspace + invalid_list)

    def test_attribute_propagation(self):
        self.set_attribute()
        new_workspace = HistogramWorkspace(self.workspace.raw_ws, 'new')
        self.check_attribute_propagation(new_workspace)
