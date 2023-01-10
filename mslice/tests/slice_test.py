from mock import MagicMock, patch
import unittest

from mslice.models.slice.slice import Slice


class SliceTest(unittest.TestCase):
    def test_sample_temp_error_if_none(self):
        test_slice = self._create_slice(sample_temp=None)
        self.assertRaises(ValueError, lambda: test_slice.sample_temp)

    def test_sample_temp_sets_and_returns(self):
        test_temp = 5.0
        test_slice = self._create_slice()
        test_slice.sample_temp = test_temp
        self.assertEqual(test_slice.sample_temp, test_temp)

    @patch('mslice.models.slice.slice.compute_chi')
    def test_chi_computes_if_none(self, compute_chi_fn):
        test_output_workspace = MagicMock()
        test_sample_temp = MagicMock()
        test_energy_axis = MagicMock()

        test_slice = self._create_slice(workspace=test_output_workspace, sample_temp=test_sample_temp,
                                        e_axis=test_energy_axis)
        test_slice.chi
        compute_chi_fn.assert_called_once_with(test_output_workspace, test_sample_temp, test_energy_axis)

    def test_chi_returns_if_already_computed(self):
        test_chi = 5.0
        test_slice = self._create_slice()
        test_slice._chi = test_chi
        self.assertEqual(test_slice.chi, test_chi)

    @patch('mslice.models.slice.slice.compute_chi')
    def test_chi__magnetic_computes_if_none(self, compute_chi_fn):
        test_output_workspace = MagicMock()
        test_sample_temp = MagicMock()
        test_energy_axis = MagicMock()

        test_slice = self._create_slice(workspace=test_output_workspace, sample_temp=test_sample_temp,
                                        e_axis=test_energy_axis)
        test_slice.chi_magnetic
        compute_chi_fn.assert_called_once_with(test_output_workspace, test_sample_temp, test_energy_axis, True)

    def test_ch_magnetic_returns_if_already_computed(self):
        test_chi_magnetic = 5.0
        test_slice = self._create_slice()
        test_slice._chi_magnetic = test_chi_magnetic
        self.assertEqual(test_slice.chi_magnetic, test_chi_magnetic)

    @patch('mslice.models.slice.slice.compute_d2sigma')
    def test_d2sigma_computes_if_none(self, compute_d2sigma_fn):
        test_output_workspace = MagicMock()
        test_energy_axis = MagicMock()

        test_slice = self._create_slice(workspace=test_output_workspace, e_axis=test_energy_axis)
        test_slice.d2sigma
        compute_d2sigma_fn.assert_called_once_with(test_output_workspace, test_energy_axis,
                                                   test_output_workspace.scattering_function.e_fixed)

    def test_d2sigma_returns_if_already_computed(self):
        test_d2sigma = 5.0
        test_slice = self._create_slice()
        test_slice._d2sigma = test_d2sigma
        self.assertEqual(test_slice.d2sigma, test_d2sigma)

    @patch('mslice.models.slice.slice.compute_symmetrised')
    def test_symmetrised_computes_if_none(self, compute_symmetrised_fn):
        test_output_workspace = MagicMock()
        test_sample_temp = MagicMock()
        test_momentum_axis = MagicMock()
        test_energy_axis = MagicMock()

        test_slice = self._create_slice(workspace=test_output_workspace, sample_temp=test_sample_temp,
                                        q_axis=test_momentum_axis, e_axis=test_energy_axis)
        test_slice.symmetrised
        compute_symmetrised_fn.assert_called_once_with(test_output_workspace, test_sample_temp,test_momentum_axis, test_energy_axis, None)

    def test_symmetrised_returns_if_already_computed(self):
        test_symmetrised = 5.0
        test_slice = self._create_slice()
        test_slice._symmetrised = test_symmetrised
        self.assertEqual(test_slice.symmetrised, test_symmetrised)

    @patch('mslice.models.slice.slice.slice_compute_gdos')
    def test_gdos_computes_if_none(self, compute_gdos_fn):
        test_output_workspace = MagicMock()
        test_sample_temp = MagicMock()
        test_energy_axis = MagicMock()

        test_slice = self._create_slice(workspace=test_output_workspace, sample_temp=test_sample_temp,
                                        e_axis=test_energy_axis)
        test_slice.gdos
        compute_gdos_fn.assert_called_once_with(test_output_workspace, test_sample_temp, None, test_energy_axis, None)

    def test_gdos_returns_if_already_computed(self):
        test_gdos = 5.0
        test_slice = self._create_slice()
        test_slice._gdos = test_gdos
        self.assertEqual(test_slice.gdos, test_gdos)

    @staticmethod
    def _create_slice(workspace=None, colourmap=None, norm=None, sample_temp=None,
                      q_axis=None, e_axis=None, rotated=None):
        return Slice(workspace, colourmap, norm, sample_temp, q_axis, e_axis, rotated)
