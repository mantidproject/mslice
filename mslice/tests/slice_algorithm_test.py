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
            return self.MockProperty(self.test_ws)
        elif args[0] == 'XAxis':
            return self.MockProperty(self.x_dict)
        elif args[0] == 'YAxis':
            return self.MockProperty(self.x_dict)
        elif args[0] == 'NormToOne':
            return self.MockProperty(self.norm_to_one)
        elif args[0] == 'PSD':
            return self.MockProperty(self.PSD)
        elif args[0] == 'EMode':
            return self.MockProperty(self.test_ws.e_mode)


    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25).transpose()
        cls.scattering_rotated = np.rot90(cls.sim_scattering_data, k=3)
        cls.scattering_rotated = np.flipud(cls.scattering_rotated)
        cls.x_dict = {'units': cls.MockProperty('DeltaE'), 'start': cls.MockProperty(-10), 'end': cls.MockProperty(15),
                      'step': cls.MockProperty(1), 'e_unit': cls.MockProperty('meV')}
        cls.y_dict = {'units': cls.MockProperty('|Q|'), 'start': cls.MockProperty(0.1), 'end': cls.MockProperty(3.1),
                      'step': cls.MockProperty(0.1), 'e_unit': cls.MockProperty('meV')}
        cls.e_axis = Axis(cls.x_dict['units'].value, cls.x_dict['start'].value, cls.x_dict['end'].value,
                          cls.x_dict['step'].value, cls.x_dict['e_unit'].value)
        cls.q_axis = Axis(cls.y_dict['units'].value, cls.y_dict['start'].value, cls.y_dict['end'].value,
                          cls.y_dict['step'].value, cls.y_dict['e_unit'].value)
        cls.q_axis_degrees = Axis('Degrees', 3, 33, 1)

        cls.test_ws = CreateSampleWorkspace(OutputWorkspace='test_ws', NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                            XMax=3.1, BinWidth=0.1, XUnit='DeltaE')
        for i in range(cls.test_ws.raw_ws.getNumberHistograms()):
            cls.test_ws.raw_ws.setY(i, cls.sim_scattering_data[i])
        AddSampleLog(workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        cls.test_ws.e_mode = 'Direct'
        cls.test_ws.e_fixed = 3


    def setUpSlice(self, norm_to_one=False, PSD=False):
        self.norm_to_one = norm_to_one
        self.PSD = PSD

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
    def test_PyExec(self, mock_get_property, mock_energy_units, mock_compute_nonPSD, mock_attribute_to_log,
                    mock_set_property):
        self.setUpSlice()
        mock_get_property.side_effect = self.property_side_effect
        mock_energy_units.return_value.factor_from_meV.return_value = 1.0
        mock_compute_nonPSD.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_energy_units.assert_called_once()
        mock_compute_nonPSD.assert_called_once()
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_compute_nonPSD.return_value)

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_nonPSD')
    @patch('mslice.models.slice.slice_algorithm.EnergyUnits')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_PSD(self, mock_get_property, mock_energy_units, mock_compute_nonPSD, mock_attribute_to_log,
                    mock_set_property):
        mock_get_property.side_effect = self.property_side_effect
        mock_energy_units.return_value.factor_from_meV.return_value = 1.0
        mock_compute_nonPSD.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_energy_units.assert_called_once()
        mock_compute_nonPSD.assert_called_once()
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_compute_nonPSD.return_value)