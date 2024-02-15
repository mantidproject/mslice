from __future__ import (absolute_import, division, print_function)

from mock import patch, MagicMock, call, ANY
import numpy as np
import unittest

from mantid.simpleapi import AddSampleLog, AnalysisDataService
from mantid.kernel import PropertyManager
from mantid.dataobjects import MDHistoWorkspace, RebinnedOutput, Workspace2D

from mslice.models.slice.slice_algorithm import Slice
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace
from tests.testhelpers.workspace_creator import create_md_workspace
from mslice.models.axis import Axis


class SliceAlgorithmTest(unittest.TestCase):

    class MockProperty:
        def __init__(self, return_value):
            self.return_value = return_value

        @property
        def value(self):
            return self.return_value

    def _property_side_effect(self, *args):
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

    @staticmethod
    def _getDimensionIndexByName_side_effect(*args, **kwargs):
        if args[0] == 'test':
            return 1.0
        elif args[0] == 'Degrees':
            return 2.0
        elif args[0] == '2Theta':
            raise RuntimeError
        elif args[0] == 'error':
            raise RuntimeError

    @classmethod
    def setUpClass(cls):
        cls.sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25).transpose()

    def tearDown(self) -> None:
        self.test_objects = None  # reset test objects
        AnalysisDataService.clear(True)

    @staticmethod
    def _create_tst_objects(sim_scattering_data, x_dict, y_dict, norm_to_one=False, PSD=False, e_mode='Direct'):
        test_ws = CreateSampleWorkspace(OutputWorkspace='slice_algo_test_ws', NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                        XMax=3.1, BinWidth=0.1, XUnit=x_dict['units'].value)
        for i in range(test_ws.raw_ws.getNumberHistograms()):
            test_ws.raw_ws.setY(i, sim_scattering_data[i])
        AddSampleLog(workspace=test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        test_ws.e_mode = e_mode
        test_ws.e_fixed = 3
        return {'workspace': test_ws, 'x_dict': x_dict, 'y_dict': y_dict, 'norm_to_one': norm_to_one, 'PSD': PSD}

    def _create_axis_dict(self, units='DeltaE', start=-10, end=15, step=1, e_unit='meV'):
        return {'units': self.MockProperty(units), 'start': self.MockProperty(start), 'end': self.MockProperty(end),
                'step': self.MockProperty(step), 'e_unit': self.MockProperty(e_unit)}

    @staticmethod
    def _generate_axis(x_dict, y_dict):
        x_axis = Axis(x_dict['units'].value, x_dict['start'].value, x_dict['end'].value, x_dict['step'].value,
                      x_dict['e_unit'].value)
        y_axis = Axis(y_dict['units'].value, y_dict['start'].value, y_dict['end'].value, y_dict['step'].value,
                      y_dict['e_unit'].value)
        return x_axis, y_axis

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
        x_dict = self._create_axis_dict()
        y_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1)
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        mock_get_property.side_effect = self._property_side_effect
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
    @patch('mslice.models.slice.slice_algorithm.ScaleX')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_nonPSD')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_nonPSD_non_meV(self, mock_get_property, mock_compute_nonPSD, mock_ScaleX, mock_attribute_to_log,
                                   mock_set_property):
        x_dict = self._create_axis_dict(e_unit='cm-1')
        y_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1, e_unit='cm-1')
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        mock_get_property.side_effect = self._property_side_effect
        mock_compute_nonPSD.return_value = MagicMock()
        mock_ScaleX.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_compute_nonPSD.assert_called_once()
        mock_ScaleX.assert_called_with(InputWorkspace=mock_compute_nonPSD.return_value, Factor=8.065544,
                                       Operation='Multiply', StoreInADS=False)
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_ScaleX.return_value)

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_PSD')
    @patch('mslice.models.slice.slice_algorithm.EnergyUnits')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_PSD(self, mock_get_property, mock_energy_units, mock_compute_PSD, mock_attribute_to_log,
                        mock_set_property):
        x_dict = self._create_axis_dict()
        y_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1)
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict, PSD=True)
        mock_get_property.side_effect = self._property_side_effect
        mock_energy_units.return_value.factor_from_meV.return_value = 1.0
        mock_compute_PSD.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_energy_units.assert_called_once()
        mock_compute_PSD.assert_called_once()
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_compute_PSD.return_value)

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_PSD')
    @patch('mslice.models.slice.slice_algorithm.TransformMD')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_PSD_non_meV(self, mock_get_property, mock_transform_MD, mock_compute_PSD, mock_attribute_to_log,
                                mock_set_property):
        x_dict = self._create_axis_dict(e_unit='cm-1')
        y_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1, e_unit='cm-1')
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict, PSD=True)
        mock_get_property.side_effect = self._property_side_effect
        mock_compute_PSD.return_value = MagicMock()
        mock_transform_MD.return_value = MagicMock()

        test_slice = Slice()
        test_slice.PyExec()
        mock_compute_PSD.assert_called_once()
        mock_transform_MD.assert_called_with(InputWorkspace=mock_compute_PSD.return_value, Scaling=[8.065544, 1.])
        mock_attribute_to_log.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_transform_MD.return_value)

    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.slice.slice_algorithm.attribute_to_log')
    @patch('mslice.models.slice.slice_algorithm.Slice._compute_slice_PSD')
    @patch('mslice.models.slice.slice_algorithm.TransformMD')
    @patch('mslice.models.slice.slice_algorithm.PythonAlgorithm.getProperty')
    def test_PyExec_PSD_non_meV_with_DeltaE_y_axis(self, mock_get_property, mock_transform_MD, mock_compute_PSD,
                                                   mock_attribute_to_log, mock_set_property):
        x_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1, e_unit='cm-1')
        y_dict = self._create_axis_dict(e_unit='cm-1')
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict, PSD=True)
        mock_get_property.side_effect = self._property_side_effect

        test_slice = Slice()
        test_slice.PyExec()
        mock_transform_MD.assert_called_with(InputWorkspace=mock_compute_PSD.return_value, Scaling=[1., 8.065544])

    def test_dimension_index_initial_success(self):
        workspace = MagicMock()
        workspace.getDimensionIndexByName.side_effect = self._getDimensionIndexByName_side_effect
        axis = MagicMock()
        axis.units = 'test'

        test_slice = Slice()
        test_slice.dimension_index(workspace, axis)
        workspace.getDimensionIndexByName.assert_called_with(axis.units)

    def test_dimension_index_secondary_success(self):
        workspace = MagicMock()
        workspace.getDimensionIndexByName.side_effect = self._getDimensionIndexByName_side_effect
        axis = MagicMock()
        axis.units = '2Theta'

        test_slice = Slice()
        test_slice.dimension_index(workspace, axis)
        calls = [call(axis.units), call('Degrees')]
        workspace.getDimensionIndexByName.assert_has_calls(calls)

    def test_dimension_index_error(self):
        workspace = MagicMock()
        workspace.getDimensionIndexByName.side_effect = self._getDimensionIndexByName_side_effect
        axis = MagicMock()
        axis.units = 'error'

        test_slice = Slice()
        self.assertRaises(RuntimeError, lambda: test_slice.dimension_index(workspace, axis))
        calls = [call(axis.units)]
        workspace.getDimensionIndexByName.assert_has_calls(calls)

    @patch('mslice.models.slice.slice_algorithm.get_number_of_steps')
    def test_compute_slice_PSD(self, mock_get_no_steps):
        workspace = create_md_workspace(2, 'slice_algo_md_ws')

        mock_x_axis = MagicMock()
        mock_x_axis.start_meV = -10
        mock_x_axis.end_meV = 10
        mock_x_axis.units = '|Q|'

        mock_y_axis = MagicMock()
        mock_y_axis.start_meV = -10
        mock_y_axis.end_meV = 10
        mock_y_axis.units = 'DeltaE'

        mock_get_no_steps.return_value = 20

        test_slice = Slice()
        computed_slice = test_slice._compute_slice_PSD(workspace, mock_x_axis, mock_y_axis, None)

        self.assertTrue(isinstance(computed_slice, MDHistoWorkspace))
        self.assertEqual(computed_slice.getSignalArray().shape, (mock_get_no_steps.return_value,
                         mock_get_no_steps.return_value))
        self.assertEqual(np.sum(computed_slice.getNumEventsArray()), workspace.getNPoints())

        x_dim = computed_slice.getXDimension()
        self.assertEqual(x_dim.name, '|Q|')
        self.assertEqual(x_dim.getMinimum(), mock_x_axis.start_meV)
        self.assertEqual(x_dim.getMaximum(), mock_x_axis.end_meV)

        y_dim = computed_slice.getYDimension()
        self.assertEqual(y_dim.name, 'DeltaE')
        self.assertEqual(y_dim.getMinimum(), mock_y_axis.start_meV)
        self.assertEqual(y_dim.getMaximum(), mock_y_axis.end_meV)

    def test_compute_slice_nonPSD_direct(self):
        x_dict = self._create_axis_dict()
        y_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1)
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        x_axis, y_axis = self._generate_axis(x_dict, y_dict)

        test_slice = Slice()
        computed_slice = test_slice._compute_slice_nonPSD(self.test_objects['workspace'].raw_ws, x_axis, y_axis,
                                                          self.test_objects['workspace'].e_mode,
                                                          self.test_objects['norm_to_one'])

        self.assertTrue(isinstance(computed_slice, RebinnedOutput))
        self.assertEqual(computed_slice.blocksize(), (x_dict['end'].value - x_dict['start'].value)
                         / x_dict['step'].value)
        self.assertEqual(computed_slice.getNumberHistograms(), (y_dict['end'].value - y_dict['start'].value)
                         / y_dict['step'].value)
        self.assertEqual(computed_slice.getNPoints(), self.test_objects['workspace'].raw_ws.getNPoints())

    @patch('mslice.models.slice.slice_algorithm.SofQW3')
    def test_compute_slice_nonPSD_indirect(self, mock_SofQW3):
        x_dict = self._create_axis_dict()
        y_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1)
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict, e_mode='Indirect')
        x_axis, y_axis = self._generate_axis(x_dict, y_dict)

        raw_ws = self.test_objects['workspace'].raw_ws
        raw_ws.run = MagicMock()
        raw_ws.run.return_value.hasProperty.return_value = True
        raw_ws.run.return_value.getProperty.return_value.value = 'test'

        test_slice = Slice()
        test_slice._compute_slice_nonPSD(raw_ws, x_axis, y_axis, self.test_objects['workspace'].e_mode,
                                         self.test_objects['norm_to_one'])
        mock_SofQW3.assert_called_once_with(InputWorkspace=ANY, QAxisBinning=ANY, EAxisBinning=ANY, EMode=ANY,
                                            StoreInADS=ANY, EFixed='test')

    def test_compute_slice_nonPSD_error_if_no_DeltaE_axis(self):
        x_dict = self._create_axis_dict(units='|Q|')
        y_dict = self._create_axis_dict(units='|Q|', start=0.1, end=3.1, step=0.1)
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        x_axis, y_axis = self._generate_axis(x_dict, y_dict)

        test_slice = Slice()
        self.assertRaises(RuntimeError, lambda: test_slice._compute_slice_nonPSD(self.test_objects['workspace'].raw_ws,
                          x_axis, y_axis, self.test_objects['workspace'].e_mode, self.test_objects['norm_to_one']))

    def test_compute_slice_nonPSD_2Theta_axes(self):
        x_dict = self._create_axis_dict()
        y_dict = self._create_axis_dict(units='2Theta', start=0.1, end=3.1, step=0.1)
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        x_axis, y_axis = self._generate_axis(x_dict, y_dict)

        test_slice = Slice()
        computed_slice = test_slice._compute_slice_nonPSD(self.test_objects['workspace'].raw_ws, x_axis, y_axis,
                                                          self.test_objects['workspace'].e_mode,
                                                          self.test_objects['norm_to_one'])

        self.assertTrue(isinstance(computed_slice, Workspace2D))
        self.assertEqual(computed_slice.getNPoints(), self.test_objects['workspace'].raw_ws.getNPoints())

        x_dim = computed_slice.getXDimension()
        self.assertEqual(x_dim.name, 'Energy transfer')
        self.assertEqual(x_dim.getMinimum(), x_dict['start'].value)
        self.assertEqual(x_dim.getMaximum(), x_dict['end'].value)

        y_dim = computed_slice.getYDimension()
        self.assertEqual(y_dim.name, 'Scattering angle')

    def test_compute_slice_nonPSD_error_if_unsupported_axes(self):
        x_dict = self._create_axis_dict()
        y_dict = self._create_axis_dict(units='invalid', start=0.1, end=3.1, step=0.1)
        self.test_objects = self._create_tst_objects(self.sim_scattering_data, x_dict, y_dict)
        x_axis, y_axis = self._generate_axis(x_dict, y_dict)

        test_slice = Slice()
        self.assertRaises(RuntimeError, lambda: test_slice._compute_slice_nonPSD(self.test_objects['workspace'].raw_ws,
                          x_axis, y_axis, self.test_objects['workspace'].e_mode, self.test_objects['norm_to_one']))
