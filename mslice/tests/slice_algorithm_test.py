from __future__ import (absolute_import, division, print_function)

from mock import patch, MagicMock
import numpy as np
import unittest

from mantid.api import AlgorithmFactory
from mantid.simpleapi import AddSampleLog, _create_algorithm_function
from mantid.kernel import PropertyManager, PropertyManagerProperty

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


class SliceAlgorithmTest(unittest.TestCase):

    class MockProperty:
        def __init__(self, return_value):
            self.return_value = return_value

        @property
        def value(self):
            return self.return_value

    def property_side_effect(self, *args, **kwargs):
        if args[0] == 'InputWorkspace':
            return self.MockProperty(self.test_objects['workspace'])
        elif args[0] == 'XAxis':
            return self.MockProperty(self.test_objects['x_dict'])
        elif args[0] == 'YAxis':
            return self.MockProperty(self.test_objects['y_dict'])
        elif args[0] == 'NormToOne':
            return self.MockProperty(self.test_objects['norm_to_one'])
        elif args[0] == 'PSD':
            return self.MockProperty(self.test_objects['PSD'])
        elif args[0] == 'EMode':
            return self.MockProperty(self.test_objects['workspace'].e_mode)

    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25).transpose()

    @staticmethod
    def create_tst_objects(sim_scattering_data, x_dict, y_dict, norm_to_one=False, PSD=False):
        scattering_rotated = np.rot90(sim_scattering_data, k=3)
        scattering_rotated = np.flipud(scattering_rotated)
        e_axis = Axis(x_dict['units'].value, x_dict['start'].value, x_dict['end'].value,
                      x_dict['step'].value, x_dict['e_unit'].value)
        q_axis = Axis(y_dict['units'].value, y_dict['start'].value, y_dict['end'].value,
                      y_dict['step'].value, y_dict['e_unit'].value)
        q_axis_degrees = Axis('Degrees', 3, 33, 1)
        test_ws = CreateSampleWorkspace(OutputWorkspace='test_ws', NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                        XMax=3.1, BinWidth=0.1, XUnit=x_dict['units'].value)
        for i in range(test_ws.raw_ws.getNumberHistograms()):
            test_ws.raw_ws.setY(i, sim_scattering_data[i])
        AddSampleLog(workspace=test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        test_ws.e_mode = 'Direct'
        test_ws.e_fixed = 3
        return {'workspace': test_ws, 'x_dict': x_dict, 'y_dict': y_dict, 'norm_to_one': norm_to_one, 'PSD': PSD}

    def create_axis_dict(self, units='DeltaE', start=-10, end=15, step=1, e_unit='meV'):
        return {'units': self.MockProperty(units), 'start': self.MockProperty(start), 'end': self.MockProperty(end),
                'step': self.MockProperty(step), 'e_unit': self.MockProperty(e_unit)}

    def test_PyInit(self):
        test_slice = Slice()
        test_slice.PyInit()
        self.assertEqual(test_slice.getProperty('InputWorkspace').value, None)
        x_axis_pmngr = test_slice.getProperty('XAxis').value
        y_axis_pmngr = test_slice.getProperty('YAxis').value
        self.assertTrue(isinstance(x_axis_pmngr, PropertyManager))
        self.assertTrue(isinstance(y_axis_pmngr, PropertyManager))
        self.assertEqual(len(x_axis_pmngr), 0)
        self.assertEqual(len(y_axis_pmngr), 0)
        self.assertEqual(test_slice.getProperty('EMode').value, 'Direct')
        self.assertEqual(test_slice.getProperty('PSD').value, False)
        self.assertEqual(test_slice.getProperty('NormToOne').value, False)
        self.assertEqual(test_slice.getProperty('OutputWorkspace').value, None)

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_nonPSD')
    @patch('mslice.models.slice.slice_algorithm.EnergyUnits')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_nonPSD(self, mock_get_property, mock_energy_units, mock_compute_nonPSD, mock_attribute_to_log,
                           mock_set_property):
        x_dict = self.create_axis_dict()
        y_dict = self.create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1)
        self.test_objects = self.create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        mock_get_property.side_effect = self.property_side_effect
        mock_energy_units.return_value.factor_from_meV.return_value = 1.0
        mock_compute_nonPSD.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_energy_units.assert_called_once()
        mock_compute_nonPSD.assert_called_once()
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_compute_nonPSD.return_value)

        self.test_objects = None  # reset test objects

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.ScaleX')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_nonPSD')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_nonPSD_non_meV(self, mock_get_property, mock_compute_nonPSD, mock_ScaleX, mock_attribute_to_log,
                    mock_set_property):
        x_dict = self.create_axis_dict(e_unit='cm-1')
        y_dict = self.create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1, e_unit='cm-1')
        self.test_objects = self.create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        mock_get_property.side_effect = self.property_side_effect
        mock_compute_nonPSD.return_value = MagicMock()
        mock_ScaleX.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_compute_nonPSD.assert_called_once()
        mock_ScaleX.assert_called_with(InputWorkspace=mock_compute_nonPSD.return_value, Factor=8.065544,
                                       Operation='Multiply', StoreInADS=False)
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_ScaleX.return_value)

        self.test_objects = None  # reset test objects

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_PSD')
    @patch('mslice.models.slice.slice_algorithm.EnergyUnits')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_PSD(self, mock_get_property, mock_energy_units, mock_compute_PSD, mock_attribute_to_log,
                    mock_set_property):
        x_dict = self.create_axis_dict()
        y_dict = self.create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1)
        self.test_objects = self.create_tst_objects(self.sim_scattering_data, x_dict, y_dict, PSD=True)
        mock_get_property.side_effect = self.property_side_effect
        mock_energy_units.return_value.factor_from_meV.return_value = 1.0
        mock_compute_PSD.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_energy_units.assert_called_once()
        mock_compute_PSD.assert_called_once()
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_compute_PSD.return_value)

        self.test_objects = None  # reset test objects

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_PSD')
    @patch('mslice.models.slice.slice_algorithm.TransformMD')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_PSD_non_meV(self, mock_get_property, mock_transform_MD, mock_compute_PSD, mock_attribute_to_log,
                                mock_set_property):
        x_dict = self.create_axis_dict(e_unit='cm-1')
        y_dict = self.create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1, e_unit='cm-1')
        self.test_objects = self.create_tst_objects(self.sim_scattering_data, x_dict, y_dict, PSD=True)
        mock_get_property.side_effect = self.property_side_effect
        mock_compute_PSD.return_value = MagicMock()
        mock_transform_MD.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_compute_PSD.assert_called_once()
        mock_transform_MD.assert_called_with(InputWorkspace=mock_compute_PSD.return_value, Scaling=[8.065544, 1.])
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_transform_MD.return_value)

        self.test_objects = None  # reset test objects

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_PSD')
    @patch('mslice.models.slice.slice_algorithm.TransformMD')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_PSD_non_meV_with_DeltaE_y_axis(self, mock_get_property, mock_transform_MD, mock_compute_PSD, mock_attribute_to_log,
                                mock_set_property):
        x_dict = self.create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1, e_unit='cm-1')
        y_dict = self.create_axis_dict(e_unit='cm-1')
        self.test_objects = self.create_tst_objects(self.sim_scattering_data, x_dict, y_dict, PSD=True)
        mock_get_property.side_effect = self.property_side_effect

        test_slice = Slice()
        test_slice.PyExec()
        mock_transform_MD.assert_called_with(InputWorkspace=mock_compute_PSD.return_value, Scaling=[1., 8.065544])

        self.test_objects = None  # reset test objects