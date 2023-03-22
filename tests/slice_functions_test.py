from __future__ import (absolute_import, division, print_function)

from mock import patch, MagicMock
import numpy as np
import unittest

from mantid.api import AlgorithmFactory
from mantid.simpleapi import AddSampleLog, _create_algorithm_function, AnalysisDataService

from mslice.models.axis import Axis
from mslice.models.slice.slice_algorithm import Slice
from mslice.models.slice.slice_functions import (compute_slice, compute_recoil_line, is_sliceable)
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace
from mslice.util.mantid.algorithm_wrapper import wrap_algorithm
from tests.testhelpers.workspace_creator import create_pixel_workspace
from mslice.models.powder.powder_functions import compute_powder_line


class SliceFunctionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25).transpose()
        cls.e_axis = Axis('DeltaE', -10, 15, 1)
        cls.q_axis = Axis('|Q|', 0.1, 3.1, 0.1)
        cls.q_axis_degrees = Axis('Degrees', 3, 33, 1)
        cls.q_axis_invalid = Axis('Invalid', 3, 33, 1)

        cls.test_ws = CreateSampleWorkspace(OutputWorkspace='slice_functions_test_ws', NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                            XMax=3.1, BinWidth=0.1, XUnit='DeltaE')
        for i in range(cls.test_ws.raw_ws.getNumberHistograms()):
            cls.test_ws.raw_ws.setY(i, cls.sim_scattering_data[i])
        AddSampleLog(workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)

    @classmethod
    def tearDownClass(cls) -> None:
        AnalysisDataService.clear()

    @patch('mslice.models.slice.slice_functions.mantid_algorithms')
    def test_slice(self, alg_mock):
        # set up slice algorithm
        AlgorithmFactory.subscribe(Slice)
        alg_mock.Slice = wrap_algorithm(_create_algorithm_function('Slice', 1, Slice()))
        plot = compute_slice('slice_functions_test_ws', self.q_axis, self.e_axis, False)
        self.assertEqual(plot.get_signal().shape, (30, 25))

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_recoil_line(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = compute_recoil_line('ws_name', self.q_axis)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.02072, 4)
        self.assertAlmostEqual(line[10], 2.5073, 4)
        self.assertAlmostEqual(line[29], 18.6491, 4)

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_recoil_line_mass(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = compute_recoil_line('ws_name', self.q_axis, 4)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.005180, 6)
        self.assertAlmostEqual(line[10], 0.626818, 6)
        self.assertAlmostEqual(line[29], 4.662281, 6)

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_recoil_line_degrees_direct(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = compute_recoil_line('ws_name', self.q_axis_degrees)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.054744, 6)
        self.assertAlmostEqual(line[10], 0.999578, 6)
        self.assertAlmostEqual(line[29], 5.276328, 6)

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_recoil_line_degrees_indirect(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Indirect'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = compute_recoil_line('ws_name', self.q_axis_degrees)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.054894, 6)
        self.assertAlmostEqual(line[10], 1.052164, 6)
        self.assertAlmostEqual(line[29], 7.167136, 6)

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_recoil_line_degrees_invalid(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        self.assertRaises(RuntimeError, lambda: compute_recoil_line('ws_name', self.q_axis_invalid))

    @patch('mslice.models.powder.powder_functions.get_workspace_handle')
    def test_powder_line(self, ws_handle_mock):
        ws_handle_mock.return_value.e_fixed = 20
        x, y = compute_powder_line('ws_name', Axis('|Q|', 0.1, 9.1, 0.1), 'Copper')
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(x[0], 3.010539, 6)
        self.assertAlmostEqual(x[10], 5.764743, 6)
        self.assertTrue(np.isnan(x[29]))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], -1)

    @patch('mslice.models.powder.powder_functions.get_workspace_handle')
    def test_powder_line_degrees(self, ws_handle_mock):
        ws_handle_mock.return_value.e_fixed = 20
        x, y = compute_powder_line('ws_name', Axis('Degrees', 3, 93, 1), 'Copper')
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(x[0], 57.9614, 4)
        self.assertAlmostEqual(x[4], 68.0383, 4)
        self.assertTrue(np.isnan(x[5]))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], -1)

    def test_is_slicable_pixel_workspace(self):
        workspace = create_pixel_workspace(2, 'slice_function_pixel_test_ws')
        self.assertTrue(is_sliceable(workspace))

    def test_is_slicable_workspace_valid(self):
        self.assertTrue(is_sliceable(self.test_ws))

    @patch('mslice.models.slice.slice_functions.WorkspaceUnitValidator')
    def test_is_slicable_workspace_invalid(self, mock_workspace_validator):
        mock_workspace_validator.return_value = MagicMock()
        mock_workspace_validator.return_value.isValid.return_value = False
        self.assertFalse(is_sliceable(self.test_ws))
