from __future__ import (absolute_import, division, print_function)

from mock import patch, MagicMock
import numpy as np
import unittest

from mslice.models.axis import Axis
from mslice.models.intensity_correction_algs import (compute_boltzmann_dist, compute_chi, compute_d2sigma,
                                                     compute_symmetrised, sample_temperature, cut_compute_gdos,
                                                     slice_compute_gdos)
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace
from mantid.simpleapi import AddSampleLog


def invert_axes(matrix):
    return np.rot90(np.flipud(matrix))


class IntensityCorrectionAlgsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.linspace(0.002, 1.5, 30*25).reshape(30, 25).transpose()
        cls.e_axis = Axis('DeltaE', cls.sim_scattering_data[0][0], cls.sim_scattering_data[24][29], 0.002 * 25)
        cls.q_axis = Axis('|Q|', cls.sim_scattering_data[0][0], cls.sim_scattering_data[24][0], 0.002)
        cls.q_axis_degrees = Axis('Degrees', 3, 33, 1)
        cls.rotated = False
        cls.norm_to_one = False
        cls.algorithm = "rebin"

        cls.test_ws = CreateSampleWorkspace(OutputWorkspace='test_ws', NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                            XMax=3.1, BinWidth=0.1, XUnit='DeltaE')
        AddSampleLog(workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        for i in range(cls.test_ws.raw_ws.getNumberHistograms()):
            cls.test_ws.raw_ws.setY(i, cls.sim_scattering_data[i])
        cls.test_ws.e_fixed = 20

    def test_boltzmann_dist(self):
        e_axis = np.arange(self.e_axis.start, self.e_axis.end, self.e_axis.step)
        boltz = compute_boltzmann_dist(10, e_axis)
        self.assertAlmostEqual(boltz[0], 0.997, 2)
        self.assertAlmostEqual(boltz[15], 0.418, 2)
        self.assertAlmostEqual(boltz[29], 0.185, 2)

    def test_compute_chi(self):
        chi = compute_chi(self.test_ws, 10, self.e_axis).get_signal()
        self.assertAlmostEqual(chi[0][0], 0.000015, 6)
        self.assertAlmostEqual(chi[15][10], 0.755691, 6)
        self.assertAlmostEqual(chi[24][29], 3.885829, 6)

    def test_compute_chi_magnetic(self):
        chi_m = compute_chi(self.test_ws, 10, self.e_axis, True).get_signal()
        self.assertAlmostEqual(chi_m[0][0], 0.0, 6)
        self.assertAlmostEqual(chi_m[15][10], 0.002597, 6)
        self.assertAlmostEqual(chi_m[24][29], 0.013353, 6)

    @patch('mslice.models.slice.slice_functions.get_workspace_handle')
    def test_compute_d2sigma(self, ws_handle_mock):
        ws_handle_mock.return_value = self.test_ws
        d2sigma = compute_d2sigma(self.test_ws, self.e_axis, self.test_ws.e_fixed).get_signal()
        self.assertAlmostEqual(d2sigma[0][0], 0.002, 6)
        self.assertAlmostEqual(d2sigma[15][10], 0.525058, 6)
        self.assertAlmostEqual(d2sigma[24][29], 1.442654, 6)

    def test_compute_symmetrised(self):
        symmetrised = compute_symmetrised(self.test_ws, 10, self.e_axis, False).get_signal()
        self.assertAlmostEqual(symmetrised[0][0], 0.002, 6)
        self.assertAlmostEqual(symmetrised[15][10], 0.532, 6)
        self.assertAlmostEqual(symmetrised[24][29], 1.5, 6)

    def test_compute_gdos_slice(self):
        gdos = slice_compute_gdos(self.test_ws, 10, self.q_axis, self.e_axis, self.rotated).get_signal()
        self.assertAlmostEqual(gdos[0][0], 0.002318, 6)
        self.assertAlmostEqual(gdos[15][10], 121.811019, 6)
        self.assertAlmostEqual(gdos[24][29], 742.138661, 6)

    @patch('mslice.models.intensity_correction_algs.compute_cut')
    @patch('mslice.models.intensity_correction_algs.slice_compute_gdos')
    @patch('mslice.models.intensity_correction_algs.get_workspace_handle')
    def test_compute_gdos_cut(self, ws_handle_mock, slice_compute_gdos_mock, compute_cut_mock):
        workspace = MagicMock()
        workspace.limits = {self.q_axis.units: [0.1, 3.1, 0.1], self.e_axis.units: [0.1, 3.1, 0.1]}
        ws_handle_mock.return_value = workspace
        slice_compute_gdos_mock.return_value = MagicMock()
        compute_cut_mock.return_value = MagicMock()
        gdos = cut_compute_gdos(self.test_ws, 10, self.q_axis, self.e_axis, self.rotated, self.norm_to_one,
                                self.algorithm)
        slice_q_axis = Axis(self.q_axis.units, 0.1, 3.1, 0.1, self.q_axis.e_unit)
        slice_e_axis = Axis(self.e_axis.units, 0.1, 3.1, 0.1, self.e_axis.e_unit)
        slice_compute_gdos_mock.assert_called_with(workspace, 10, slice_q_axis, slice_e_axis, self.rotated)
        compute_cut_mock.assert_called_with(slice_compute_gdos_mock.return_value, self.q_axis, self.e_axis,
                                            self.norm_to_one, self.algorithm)
        self.assertEqual(gdos, compute_cut_mock.return_value)

    def test_sample_temperature(self):
        self.assertEqual(sample_temperature(self.test_ws.name, ['Ei']), 3.0)

    def test_sample_temperature_string(self):
        AddSampleLog(workspace=self.test_ws.raw_ws, LogName='StrTemp', LogText='3.', LogType='String', StoreInADS=False)
        self.assertEqual(sample_temperature(self.test_ws.name, ['StrTemp']), '3.')

    def test_sample_temperature_list(self):
        AddSampleLog(workspace=self.test_ws.raw_ws, LogName='ListTemp', LogText='3.', LogType='Number Series',
                     StoreInADS=False)
        self.assertEqual(sample_temperature(self.test_ws.name, ['ListTemp']), [3.0])
