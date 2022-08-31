from __future__ import (absolute_import, division, print_function)

from mock import patch
import numpy as np
import unittest

from mslice.models.axis import Axis
from mslice.models.intensity_correction_algs import (compute_boltzmann_dist, compute_chi, compute_d2sigma,
                                                     compute_symmetrised, sample_temperature)
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace
from mantid.simpleapi import AddSampleLog


def invert_axes(matrix):
    return np.rot90(np.flipud(matrix))


class IntensityCorrectionAlgsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30,25).transpose()
        cls.e_axis = Axis('DeltaE', -10, 15, 1)
        cls.q_axis = Axis('|Q|', 0.1, 3.1, 0.1)
        cls.q_axis_degrees = Axis('Degrees', 3, 33, 1)

        cls.test_ws = CreateSampleWorkspace(OutputWorkspace='test_ws', NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                            XMax=3.1, BinWidth=0.1, XUnit='DeltaE')
        AddSampleLog(workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        for i in range(cls.test_ws.raw_ws.getNumberHistograms()):
            cls.test_ws.raw_ws.setY(i, cls.sim_scattering_data[i])
        cls.test_ws.e_fixed = 20

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
        chi_m = compute_chi(self.test_ws, 10, self.e_axis, True).get_signal()
        self.assertAlmostEqual(chi_m[0][0], 0.0, 6)
        self.assertAlmostEqual(chi_m[15][10], 0.005713, 6)
        self.assertAlmostEqual(chi_m[24][29], 0.016172, 6)

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_compute_d2sigma(self, ws_handle_mock):
        ws_handle_mock.return_value = self.test_ws
        d2sigma = compute_d2sigma(self.test_ws, self.e_axis, self.test_ws.e_fixed).get_signal()
        self.assertAlmostEqual(d2sigma[0][0], 0.0, 6)
        self.assertAlmostEqual(d2sigma[15][10], 0.449329, 6)
        self.assertAlmostEqual(d2sigma[24][29], 0.749, 6)

    def test_compute_symmetrised(self):
        symmetrised = compute_symmetrised(self.test_ws, 10, self.e_axis, False).get_signal()
        self.assertAlmostEqual(symmetrised[0][0], 0.0, 6)
        self.assertAlmostEqual(symmetrised[15][10], 0.53, 6)
        self.assertAlmostEqual(symmetrised[24][29], 1.498, 6)

    def test_sample_temperature(self):
        self.assertEqual(sample_temperature(self.test_ws.name, ['Ei']), 3.0)

    def test_sample_temperature_string(self):
        AddSampleLog(workspace=self.test_ws.raw_ws, LogName='StrTemp', LogText='3.', LogType='String', StoreInADS=False)
        self.assertEqual(sample_temperature(self.test_ws.name, ['StrTemp']), '3.')

    def test_sample_temperature_list(self):
        AddSampleLog(workspace=self.test_ws.raw_ws, LogName='ListTemp', LogText='3.', LogType='Number Series',
                     StoreInADS=False)
        self.assertEqual(sample_temperature(self.test_ws.name, ['ListTemp']), [3.0])
