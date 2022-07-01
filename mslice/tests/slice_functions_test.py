from __future__ import (absolute_import, division, print_function)

from mock import patch
import numpy as np
import unittest

from mantid.api import AlgorithmFactory
from mantid.simpleapi import AddSampleLog, _create_algorithm_function, AnalysisDataService

from mslice.models.axis import Axis
from mslice.models.slice.slice_algorithm import Slice
from mslice.models.slice.slice_functions import (compute_slice, compute_boltzmann_dist, compute_chi,
                                                 compute_chi_magnetic, compute_d2sigma, compute_symmetrised,
                                                 compute_gdos, compute_recoil_line)
from mslice.models.powder.powder_functions import compute_powder_line
from mslice.util.mantid.algorithm_wrapper import wrap_algorithm
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace


def invert_axes(matrix):
    return np.rot90(np.flipud(matrix))


class SliceFunctionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30,25).transpose()
        cls.scattering_rotated = np.rot90(cls.sim_scattering_data, k=3)
        cls.scattering_rotated = np.flipud(cls.scattering_rotated)
        cls.e_axis = Axis('DeltaE', -10, 15, 1)
        cls.q_axis = Axis('|Q|', 0.1, 3.1, 0.1)
        cls.q_axis_degrees = Axis('Degrees', 3, 33, 1)

        cls.test_ws = CreateSampleWorkspace(OutputWorkspace='test_ws', NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                            XMax=3.1, BinWidth=0.1, XUnit='DeltaE', StoreInADS=False)
        for i in range(cls.test_ws.raw_ws.getNumberHistograms()):
            cls.test_ws.raw_ws.setY(i, cls.sim_scattering_data[i])
        AddSampleLog(workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        cls.test_ws.e_mode = 'Direct'
        cls.test_ws.e_fixed = 3

    @patch('mslice.models.slice.slice_functions.mantid_algorithms')
    def test_slice(self, alg_mock):
        # set up slice algorithm
        AlgorithmFactory.subscribe(Slice)
        alg_mock.Slice = wrap_algorithm(_create_algorithm_function('Slice', 1, Slice()))
        plot = compute_slice('test_ws', self.q_axis, self.e_axis, False)
        self.assertEqual(plot.get_signal().shape, (30, 25))

    def test_boltzmann_dist(self):
        e_axis = np.arange(self.e_axis.start, self.e_axis.end, self.e_axis.step)
        boltz = compute_boltzmann_dist(10, e_axis)
        self.assertAlmostEqual(boltz[0], 109592.269, 0)
        self.assertAlmostEqual(boltz[10], 1.0, 3)
        self.assertAlmostEqual(boltz[20], 0.000009125, 9)

    def test_compute_chi(self):
        chi = compute_chi(self.test_ws, 10, self.e_axis).get_signal()
        self.assertAlmostEqual(chi[0][0], 0.0, 6)
        self.assertAlmostEqual(chi[15][10], 1.662609, 6)
        self.assertAlmostEqual(chi[24][29], 4.706106, 6)

    def test_compute_chi_magnetic(self):
        chi = compute_chi(self.test_ws, 10, self.e_axis).get_signal()
        chi_m = compute_chi_magnetic(chi)
        self.assertAlmostEqual(chi_m[0][0], 0.0, 6)
        self.assertAlmostEqual(chi_m[15][10], 0.005713, 6)
        self.assertAlmostEqual(chi_m[24][29], 0.016172, 6)

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_compute_d2sigma(self, ws_handle_mock):
        ws_handle_mock.return_value = self.test_ws
        self.test_ws.e_fixed = 20
        d2sigma = compute_d2sigma(self.test_ws, self.e_axis).get_signal()
        self.assertAlmostEqual(d2sigma[0][0], 0.0, 6)
        self.assertAlmostEqual(d2sigma[15][10], 0.449329, 6)
        self.assertAlmostEqual(d2sigma[24][29], 0.749, 6)
        self.test_ws.e_fixed = 20

    def test_compute_symmetrised(self):
        symmetrised = compute_symmetrised(self.test_ws, 10, self.e_axis, False).get_signal()
        self.assertAlmostEqual(symmetrised[0][0], 0.0, 6)
        self.assertAlmostEqual(symmetrised[15][10], 0.53, 6)
        self.assertAlmostEqual(symmetrised[24][29], 1.498, 6)

    def test_compute_gdos(self):
        gdos = compute_gdos(self.test_ws, 10, self.q_axis, self.e_axis).get_signal()
        self.assertAlmostEqual(gdos[0][0], 0.0, 6)
        self.assertAlmostEqual(gdos[15][10], 2.312954, 6)
        self.assertAlmostEqual(gdos[24][29], 2.338189, 6)

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
    def test_recoil_line_degrees(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = compute_recoil_line('ws_name', self.q_axis_degrees)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.054744, 6)
        self.assertAlmostEqual(line[10], 0.999578, 6)
        self.assertAlmostEqual(line[29], 5.276328, 6)

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
