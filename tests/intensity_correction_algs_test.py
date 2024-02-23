from __future__ import (absolute_import, division, print_function)

from mock import patch, MagicMock, call
import numpy as np
import unittest
from copy import copy

from mslice.models.axis import Axis
from mslice.models.intensity_correction_algs import (compute_boltzmann_dist, compute_chi, compute_d2sigma,
                                                     compute_symmetrised, sample_temperature, cut_compute_gdos,
                                                     slice_compute_gdos, _cut_compute_gdos, _cut_compute_gdos_pixel,
                                                     _get_slice_axis, _reduce_bins_along_int_axis)
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace, CreateMDHistoWorkspace
from mslice.workspace.pixel_workspace import PixelWorkspace
from mantid.simpleapi import AddSampleLog


def _invert_axes(matrix):
    return np.rot90(np.flipud(matrix))


def _tag_and_return_mock(mock_obj):
    mock_obj.processed = True
    return mock_obj


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
        cls.test_ws.parent = 'parent_test_ws'
        AddSampleLog(workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        for i in range(cls.test_ws.raw_ws.getNumberHistograms()):
            cls.test_ws.raw_ws.setY(i, cls.sim_scattering_data[i])
        cls.test_ws.e_fixed = 20
        cls.test_ws.raw_ws.getAxis(1).setUnit('MomentumTransfer')

        cls.test_psd = CreateMDHistoWorkspace(OutputWorkspace='test_psd', Dimensionality=2, Extents=[0.1, 3.1, 1., 25.],
                                              NumberOfBins=[25, 30], Names='Energy transfer,q', Units='meV,Angstrom^-1',
                                              SignalInput=cls.test_ws.get_signal(), ErrorInput=cls.test_ws.get_error())
        cls.test_psd.axes = (cls.e_axis, cls.q_axis)
        cls.test_psd.name = 'test_psd'
        cls.test_psd.parent = 'parent_test_psd'

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

    def test_compute_symmetrised_rotated_psd(self):
        self.test_ws.is_PSD = True
        symmetrised = compute_symmetrised(self.test_ws, 10, self.e_axis, True).get_signal()
        self.assertAlmostEqual(symmetrised[0][0], 0.002, 6)
        self.assertAlmostEqual(symmetrised[15][10], 0.532, 6)
        self.assertAlmostEqual(symmetrised[24][29], 1.5, 6)
        self.test_ws.is_PSD = None  # reset workspace for other tests.

    def test_compute_gdos_slice(self):
        gdos = slice_compute_gdos(self.test_ws, 10, self.q_axis, self.e_axis, self.rotated).get_signal()
        self.assertAlmostEqual(gdos[0][0], 0.002318, 6)
        self.assertAlmostEqual(gdos[15][10], 121.811019, 6)
        self.assertAlmostEqual(gdos[24][29], 742.138661, 6)

    @patch('mslice.models.intensity_correction_algs.PixelWorkspace.__init__')
    @patch('mslice.models.intensity_correction_algs._cut_compute_gdos_pixel')
    @patch('mslice.models.intensity_correction_algs.get_workspace_handle')
    def test_cut_compute_gdos_pixel(self, ws_handle_mock, cut_compute_gdos_pixel_mock, pixel_ws_init_mock):
        pixel_ws_init_mock.return_value = None
        workspace = PixelWorkspace()
        workspace._histo_ws = MagicMock()  # avoid error with destructor
        workspace._raw_ws = MagicMock()  # avoid error with destructor
        ws_handle_mock.return_value = workspace
        cut_compute_gdos(self.test_ws, 10, self.q_axis, self.e_axis, self.rotated, self.norm_to_one,
                         self.algorithm, True)
        cut_compute_gdos_pixel_mock.assert_called_once()

    def test_cut_compute_gdos_impl(self):
        self._internal_tst_cut_compute_gdos_impl(_cut_compute_gdos, [call(self.test_ws.parent)])

    def test_cut_compute_gdos_impl_pixel(self):
        self._internal_tst_cut_compute_gdos_impl(_cut_compute_gdos_pixel,
                                                 [call(self.test_ws.parent), call('__' + self.test_ws.parent)])

    @patch('mslice.models.intensity_correction_algs.slice_compute_gdos')
    @patch('mslice.models.intensity_correction_algs._reduce_bins_along_int_axis')
    @patch('mslice.models.intensity_correction_algs.compute_slice')
    @patch('mslice.models.intensity_correction_algs._get_slice_axis')
    @patch('mslice.models.intensity_correction_algs.get_workspace_handle')
    def _internal_tst_cut_compute_gdos_impl(self, test_fn, ws_handle_calls, ws_handle_mock, get_slice_axis_mock, compute_slice_mock,
                                            reduce_bins_mock, slice_compute_gdos_mock):
        parent_workspace = MagicMock()
        parent_workspace.limits = {self.q_axis.units: [0.1, 3.1, 0.1], self.e_axis.units: [0.1, 6.2, 0.2]}
        x_dim_mock = MagicMock()
        y_dim_mock = MagicMock()
        x_dim_mock.getUnits.return_value = self.q_axis.units
        y_dim_mock.getUnits.return_value = self.e_axis.units
        parent_workspace._raw_ws.getXDimension.return_value = x_dim_mock
        parent_workspace.raw_ws.getXDimension.return_value = x_dim_mock
        parent_workspace._raw_ws.getYDimension.return_value = y_dim_mock
        ws_handle_mock.return_value = parent_workspace

        get_slice_axis_mock.side_effect = lambda a, b, c: b
        compute_slice_mock.side_effect = lambda a, b, c, d, store_in_ADS: _tag_and_return_mock(a)
        slice_compute_gdos_mock.side_effect = lambda a, b, c, d, e: slice_compute_gdos(a, b, c, d, e)

        test_fn(self.test_ws, 10, self.q_axis, self.e_axis, self.rotated, self.norm_to_one, self.algorithm, True)
        for mock_call in ws_handle_calls:
            ws_handle_mock.assert_has_calls([mock_call])
        self.assertEqual(2, get_slice_axis_mock.call_count)
        compute_slice_mock.assert_called_once()
        slice_compute_gdos_mock.assert_called_once_with(parent_workspace, 10, self.q_axis, self.e_axis, self.rotated)
        self.assertTrue(slice_compute_gdos_mock.call_args.args[0].processed)

        # parent_workspace.__truediv__().__imul__().__imul__() = workspace that has been divided then multiplied twice
        reduce_bins_mock.assert_called_once_with(parent_workspace.__truediv__().__imul__().__imul__(), self.algorithm, self.q_axis,
                                                 self.e_axis, 1, True, "test_ws")

    @patch('mslice.models.intensity_correction_algs._cut_compute_gdos')
    @patch('mslice.models.intensity_correction_algs.get_workspace_handle')
    def test_cut_compute_gdos(self, ws_handle_mock, cut_compute_gdos_mock):
        workspace = MagicMock()
        ws_handle_mock.return_value = workspace

        cut_compute_gdos(self.test_ws, 10, self.q_axis, self.e_axis, self.rotated, self.norm_to_one,
                         self.algorithm, True)
        cut_compute_gdos_mock.assert_called_once()

    def test_get_slice_axis_enforces_data_minimum_step_if_input_smaller(self):
        data_limits = [0.002, 0.005, 0.004]
        ret_axis = _get_slice_axis(data_limits, self.q_axis, False)
        self.assertEqual(Axis(self.q_axis.units, self.q_axis.start, self.q_axis.end, data_limits[2], self.q_axis.e_unit), ret_axis)

    def test_get_slice_axis_takes_input_step_if_larger(self):
        data_limits = [0.002, 0.005, 0.001]
        ret_axis = _get_slice_axis(data_limits, self.q_axis, False)
        self.assertEqual(Axis(self.q_axis.units, self.q_axis.start, self.q_axis.end, self.q_axis.step, self.q_axis.e_unit), ret_axis)

    def test_get_slice_axis_icut_alligns_bins(self):
        data_limits = [-0.4795, 0.4795, self.q_axis.step]
        ret_axis = _get_slice_axis(data_limits, self.q_axis, True)
        self.assertEqual(Axis(self.q_axis.units, 0.0005, 0.0505, self.q_axis.step, self.q_axis.e_unit), ret_axis)

    def test_reduce_bins_cut(self):
        self._internal_tst_reduce_bins('Rebin', False, False)

    def test_reduce_bins_integration(self):
        self._internal_tst_reduce_bins('Integration', False, False)

    def test_reduce_bins_cut_rotated(self):
        self._internal_tst_reduce_bins('Rebin', True, False)

    def test_reduce_bins_cut_psd(self):
        self._internal_tst_reduce_bins('Rebin', False, True)

    def test_reduce_bins_integration_psd(self):
        self._internal_tst_reduce_bins('Integration', False, True)

    def test_reduce_bins_cut_rotated_psd(self):
        self._internal_tst_reduce_bins('Rebin', True, True)

    @patch('mslice.models.intensity_correction_algs._cut_nonPSD_general')
    @patch('mslice.models.intensity_correction_algs.CreateMDHistoWorkspace')
    def _internal_tst_reduce_bins(self, algorithm, rotated, use_psd, createMDHistoWorkspace_mock, cutalgo_mock):
        if rotated:
            units = 'meV,Angstrom^-1'
            names = 'Energy transfer,q'
            cut_axis = self.e_axis
            int_axis = self.q_axis
            cut_x_dim = self.test_ws.raw_ws.getYDimension()
            cut_y_dim = self.test_ws.raw_ws.getXDimension()
            axstr = f'{int_axis.start}, {int_axis.end - int_axis.start}, {int_axis.end}'
        else:
            units = 'Angstrom^-1,meV'
            names = 'q,Energy transfer'
            cut_axis = self.q_axis
            int_axis = self.e_axis
            cut_x_dim = self.test_ws.raw_ws.getXDimension()
            cut_y_dim = self.test_ws.raw_ws.getYDimension()
            axstr = f'{cut_axis.start}, {cut_axis.step}, {cut_axis.end}'

        selected_test_ws = self.test_psd if use_psd else self.test_ws
        _reduce_bins_along_int_axis(selected_test_ws, algorithm, cut_axis, copy(int_axis), int(rotated), True, "test")

        createMDHistoWorkspace_mock.assert_called_once()
        self.assertEqual(2, createMDHistoWorkspace_mock.call_args[1]['Dimensionality'])
        self.assertEqual(f'{cut_x_dim.getMinimum()},{cut_x_dim.getMaximum()},{cut_y_dim.getMinimum()},{cut_y_dim.getMaximum()}',
                         createMDHistoWorkspace_mock.call_args[1]['Extents'])
        self.assertEqual(names, createMDHistoWorkspace_mock.call_args[1]['Names'])
        self.assertEqual(units, createMDHistoWorkspace_mock.call_args[1]['Units'])
        if use_psd:
            integration_factor = int_axis.step if algorithm == 'Integration' else 1
            signal_result = np.nansum(selected_test_ws.get_signal(), int(rotated), keepdims=True) * integration_factor
            error_result = np.sqrt(np.nansum(selected_test_ws.get_variance(), int(rotated), keepdims=True)) * integration_factor
            shape_string = f'{signal_result.shape[1]},{signal_result.shape[0]}' if rotated else f'{signal_result.shape[0]},' \
                                                                                                f'{signal_result.shape[1]}'
            self.assertEqual(shape_string, createMDHistoWorkspace_mock.call_args[1]['NumberOfBins'])
            self.assertTrue(np.allclose(signal_result, createMDHistoWorkspace_mock.call_args[1]['SignalInput'], atol=1e-08))
            self.assertTrue(np.allclose(error_result, createMDHistoWorkspace_mock.call_args[1]['ErrorInput'], atol=1e-08))
        else:
            self.assertEqual(axstr, cutalgo_mock.call_args[0][0])

    def test_sample_temperature(self):
        self.assertEqual(sample_temperature(self.test_ws.name, ['Ei']), 3.0)

    def test_sample_temperature_string(self):
        AddSampleLog(workspace=self.test_ws.raw_ws, LogName='StrTemp', LogText='3.', LogType='String', StoreInADS=False)
        self.assertEqual(sample_temperature(self.test_ws.name, ['StrTemp']), '3.')

    def test_sample_temperature_list(self):
        AddSampleLog(workspace=self.test_ws.raw_ws, LogName='ListTemp', LogText='3.', LogType='Number Series',
                     StoreInADS=False)
        print("listtemp")
        print(sample_temperature(self.test_ws.name, ['ListTemp']))
        self.assertEqual(sample_temperature(self.test_ws.name, ['ListTemp']), [3.0])
