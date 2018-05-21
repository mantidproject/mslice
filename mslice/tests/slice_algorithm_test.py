from __future__ import (absolute_import, division, print_function)

from mock import patch
import numpy as np
import unittest
from mantid.api import AlgorithmFactory
from mantid.simpleapi import AddSampleLog, _create_algorithm_function
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace
from mslice.models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
from mslice.models.axis import Axis
from mslice.models.slice.slice import Slice
from mslice.util.mantid.algorithm_wrapper import wrap_algorithm

def invert_axes(matrix):
    return np.rot90(np.flipud(matrix))


class SliceAlgorithmTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25)
        cls.scattering_rotated = np.rot90(cls.sim_scattering_data, k=3)
        cls.scattering_rotated = np.flipud(cls.scattering_rotated)
        cls.slice_alg = MantidSliceAlgorithm()
        cls.e_axis = Axis('DeltaE', -10, 15, 1)
        cls.q_axis = Axis('|Q|', 0.1, 3.1, 0.1)
        cls.q_axis_degrees = Axis('Degrees', 3, 33, 1)

        cls.test_ws = CreateSampleWorkspace(OutputWorkspace='test_ws', XUnit='DeltaE')
        AddSampleLog(Workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.',
                     LogType='Number', StoreInADS=False)
        cls.test_ws.e_mode = 'Direct'
        cls.test_ws.e_fixed = 3

    @patch('mslice.models.slice.mantid_slice_algorithm.mantid_algorithms')
    def test_slice(self, alg_mock):
        # set up slice algorithm
        AlgorithmFactory.subscribe(Slice)
        alg_mock.Slice = wrap_algorithm(_create_algorithm_function('Slice', 1, Slice()))

        plot, boundaries = self.slice_alg.compute_slice('test_ws', self.q_axis, self.e_axis, False)
        self.assertEqual(np.shape(plot[0]), (25, 30))
        np.testing.assert_array_equal(boundaries, [0.1, 3.1, -10, 15])

    def test_boltzmann_dist(self):
        e_axis = np.arange(self.e_axis.start, self.e_axis.end, self.e_axis.step)
        boltz = self.slice_alg.compute_boltzmann_dist(10, e_axis)
        self.assertAlmostEqual(boltz[0], 109592.269, 3)
        self.assertAlmostEqual(boltz[10], 1.0, 3)
        self.assertAlmostEqual(boltz[20], 0.000009125, 9)

    def test_compute_chi(self):
        chi = self.slice_alg.compute_chi(self.sim_scattering_data, 10, self.e_axis, True)
        chi_rotated = self.slice_alg.compute_chi(self.scattering_rotated, 10, self.e_axis, False)
        np.testing.assert_array_almost_equal(chi, invert_axes(chi_rotated), 6)
        self.assertAlmostEqual(chi[0][0], 0.0, 6)
        self.assertAlmostEqual(chi[10][15], 1.662609, 6)
        self.assertAlmostEqual(chi[29][24], 4.706106, 6)

    def test_compute_chi_magnetic(self):
        chi = self.slice_alg.compute_chi(self.sim_scattering_data, 10, self.e_axis, True)
        chi_m = self.slice_alg.compute_chi_magnetic(chi)
        self.assertAlmostEqual(chi_m[0][0], 0.0, 6)
        self.assertAlmostEqual(chi_m[10][15], 0.005713, 6)
        self.assertAlmostEqual(chi_m[29][24], 0.016172, 6)

    @patch('mslice.models.slice.mantid_slice_algorithm.get_workspace_handle')
    def test_compute_d2sigma(self, ws_handle_mock):
        ws_handle_mock.return_value.e_fixed = 20
        d2sigma = self.slice_alg.compute_d2sigma(self.sim_scattering_data, None, self.e_axis, True)
        d2sigma_rotated = self.slice_alg.compute_d2sigma(self.scattering_rotated, None, self.e_axis, False)
        np.testing.assert_array_almost_equal(d2sigma, invert_axes(d2sigma_rotated), 6)
        self.assertAlmostEqual(d2sigma[0][0], 0.0, 6)
        self.assertAlmostEqual(d2sigma[10][15], 0.449329, 6)
        self.assertAlmostEqual(d2sigma[29][24], 0.749, 6)

    def test_compute_symmetrised(self):
        symmetrised = self.slice_alg.compute_symmetrised(self.sim_scattering_data, 10, self.e_axis, True)
        symmetrised_rotated = self.slice_alg.compute_symmetrised(self.scattering_rotated, 10, self.e_axis, False)
        np.testing.assert_array_almost_equal(symmetrised, invert_axes(symmetrised_rotated), 6)
        self.assertAlmostEqual(symmetrised[0][0], 0.0, 6)
        self.assertAlmostEqual(symmetrised[10][15], 0.53, 6)
        self.assertAlmostEqual(symmetrised[29][24], 1.498, 6)

    def test_compute_gdos(self):
        gdos = self.slice_alg.compute_gdos(self.sim_scattering_data, 10, self.q_axis, self.e_axis, True)
        gdos_rotated = self.slice_alg.compute_gdos(self.scattering_rotated, 10, self.q_axis, self.e_axis, False)
        np.testing.assert_array_almost_equal(gdos, invert_axes(gdos_rotated), 6)
        self.assertAlmostEqual(gdos[0][0], 0.0, 6)
        self.assertAlmostEqual(gdos[10][15], 0.697758, 6)
        self.assertAlmostEqual(gdos[29][24], 2246.999938, 6)

    @patch('mslice.models.slice.mantid_slice_algorithm.get_workspace_handle')
    def test_recoil_line(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = self.slice_alg.compute_recoil_line('ws_name', self.q_axis)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.02072, 4)
        self.assertAlmostEqual(line[10], 2.5073, 4)
        self.assertAlmostEqual(line[29], 18.6491, 4)

    @patch('mslice.models.slice.mantid_slice_algorithm.get_workspace_handle')
    def test_recoil_line_mass(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = self.slice_alg.compute_recoil_line('ws_name', self.q_axis, 4)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.005180, 6)
        self.assertAlmostEqual(line[10], 0.626818, 6)
        self.assertAlmostEqual(line[29], 4.662281, 6)

    @patch('mslice.models.slice.mantid_slice_algorithm.get_workspace_handle')
    def test_recoil_line_degrees(self, ws_handle_mock):
        ws_handle_mock.return_value.e_mode = 'Direct'
        ws_handle_mock.return_value.e_fixed = 20
        x_axis, line = self.slice_alg.compute_recoil_line('ws_name', self.q_axis_degrees)
        self.assertEqual(len(line), 30)
        self.assertAlmostEqual(line[0], 0.054744, 6)
        self.assertAlmostEqual(line[10], 0.999578, 6)
        self.assertAlmostEqual(line[29], 5.276328, 6)

    @patch('mslice.models.slice.mantid_slice_algorithm.get_workspace_handle')
    def test_powder_line(self, ws_handle_mock):
        from mslice.models.axis import Axis
        ws_handle_mock.return_value.e_fixed = 20
        x, y = self.slice_alg.compute_powder_line('ws_name', Axis('|Q|', 0.1, 9.1, 0.1), 'Copper')
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(x[0], 3.010539, 6)
        self.assertAlmostEqual(x[10], 5.764743, 6)
        self.assertTrue(np.isnan(x[29]))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], -1)

    @patch('mslice.models.slice.mantid_slice_algorithm.get_workspace_handle')
    def test_powder_line_degrees(self, ws_handle_mock):
        from mslice.models.axis import Axis
        ws_handle_mock.return_value.e_fixed = 20
        x, y = self.slice_alg.compute_powder_line('ws_name', Axis('Degrees', 3, 93, 1), 'Copper')
        self.assertEqual(len(x), len(y))
        self.assertAlmostEqual(x[0], 57.9614, 4)
        self.assertAlmostEqual(x[4], 68.0383, 4)
        self.assertTrue(np.isnan(x[5]))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], -1)
