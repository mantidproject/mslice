from __future__ import (absolute_import, division, print_function)

from mock import patch
import numpy as np
import unittest

from mslice.models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
from mslice.presenters.slice_plotter_presenter import Axis


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

    def test_boltzmann_dist(self):
        e_axis = np.arange(self.e_axis.start, self.e_axis.end, self.e_axis.step)
        boltz = self.slice_alg.compute_boltzmann_dist(10, e_axis)
        self.assertAlmostEqual(boltz[0], 109591.959, 3)
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

    @patch('mslice.models.slice.mantid_slice_algorithm.MantidWorkspaceProvider.get_EFixed')
    @patch('mslice.models.slice.mantid_slice_algorithm.MantidWorkspaceProvider.get_workspace_handle')
    def test_compute_d2sigma(self, ws_handle_mock, get_efixed_mock):
        get_efixed_mock.return_value = 20
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
