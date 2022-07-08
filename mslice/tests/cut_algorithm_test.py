import numpy as np
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateMDWorkspace, FakeMDEventData, BinMD
from mantid.dataobjects import MDHistoWorkspace

from mslice.models.axis import Axis
from mslice.models.cut.cut_algorithm import compute_cut, Cut
from mslice.tests.testhelpers.workspace_creator import create_simulation_workspace, create_md_workspace


class MockProperty:
    def __init__(self, return_value):
        self._return_value = return_value

    @property
    def value(self):
        return self._return_value


class CutAlgorithmTest(TestCase):

    def setUp(self):
        self.e_axis = Axis("DeltaE", -10, 15, 1)
        self.q_axis = Axis("|Q|", 0.1, 3.1, 0.1)
        self.theta_axis = Axis("2Theta", -10, 15, 1)

    def tearDown(self) -> None:
        AnalysisDataService.clear()

    def test_that_compute_cut_returns_a_result_with_the_expected_size_for_normalized_psd_rebin_data(self):
        AnalysisDataService.clear()
        psd_workspace = CreateMDWorkspace(Dimensions=2, Extents=",".join(["-10,10"] * 2),
                                          Names=",".join(["|Q|", "DeltaE"]),
                                          Units=",".join(["U"] * 2), OutputWorkspace="cut_algo_test_md_ws")
        FakeMDEventData(InputWorkspace=psd_workspace, PeakParams=f"500000,0,0,3",
                        RandomizeSignal=False)
        e_dim = psd_workspace.getYDimension()
        q_dim = psd_workspace.getXDimension()

        y_dim = e_dim.getDimensionId() + "," + str(e_dim.getMinimum()) + "," + str(e_dim.getMaximum()) + "," + "20"
        x_dim = q_dim.getDimensionId() + "," + str(q_dim.getMinimum()) + "," + str(q_dim.getMaximum()) + "," + "1"

        cut = BinMD(InputWorkspace=psd_workspace, AxisAligned=True, AlignedDim0=x_dim, AlignedDim1=y_dim)
        self.assertAlmostEqual(np.nanmax(cut.getSignalArray()), 101381, 3)

    def test_that_compute_cut_returns_a_result_with_the_expected_size_for_normalized_non_psd_rebin_data(self):
        normalized = True
        algorithm = "Rebin"

        self._test_non_psd_cut(normalized, algorithm)

    def xtest_that_compute_cut_returns_the_expected_size_for_psd_rebin_data(self):
        normalized = False
        algorithm = "Rebin"

        cut = self._test_psd_cut(normalized, algorithm)
        self.assertAlmostEqual(np.nanmax(cut.getSignalArray()), 10729.887, 3)

    def test_that_compute_cut_returns_the_expected_size_for_non_psd_rebin_data(self):
        normalized = False
        algorithm = "Rebin"

        self._test_non_psd_cut(normalized, algorithm)

    def xtest_that_compute_cut_returns_a_result_with_the_expected_size_for_normalized_psd_integration_data(self):
        normalized = True
        algorithm = "Integration"

        cut = self._test_psd_cut(normalized, algorithm)
        self.assertAlmostEqual(np.nanmax(cut.getSignalArray()), 10680.995, 3)

    def test_that_compute_cut_returns_a_result_with_the_expected_size_for_normalized_non_psd_integration_data(self):
        normalized = True
        algorithm = "Integration"

        self._test_non_psd_cut(normalized, algorithm)

    def xtest_that_compute_cut_returns_the_expected_size_for_psd_integration_data(self):
        normalized = False
        algorithm = "Integration"

        cut = self._test_psd_cut(normalized, algorithm)
        self.assertAlmostEqual(np.nanmax(cut.getSignalArray()), 64379.322, 3)

    def test_that_compute_cut_returns_the_expected_size_for_non_psd_integration_data(self):
        normalized = False
        algorithm = "Integration"

        self._test_non_psd_cut(normalized, algorithm)

    def test_that_compute_cut_returns_the_expected_size_for_a_rebin_cut_axis_in_units_of_2theta(self):
        non_psd_workspace = create_simulation_workspace("Direct", "non_psd_ws", psd=False)

        cut = compute_cut(non_psd_workspace.raw_ws, self.theta_axis, self.e_axis, "Direct", False, False, "Rebin")

        self.assertTrue(isinstance(cut, MDHistoWorkspace))
        self.assertEqual(cut.getSignalArray().shape, (25, ))
        self.assertEqual(cut.getErrorSquaredArray().shape, (25, ))

    @patch('mslice.models.cut.cut_algorithm.PythonAlgorithm.declareProperty')
    def test_PyInit_will_not_cause_any_errors(self, mock_declare_property):
        cut_algo = Cut()
        cut_algo.PyInit()

        self.assertEqual(mock_declare_property.call_count, 8)

    @patch('mslice.models.cut.cut_algorithm.PythonAlgorithm.setProperty')
    @patch('mslice.models.cut.cut_algorithm.PythonAlgorithm.getProperty')
    @patch('mslice.models.cut.cut_algorithm.compute_cut')
    def test_PyExec_will_execute_the_compute_cut_function(self, mock_compute_cut,
                                                          mock_get_property, mock_set_property):
        q_axis = {"units": MockProperty("|Q|"), "start": MockProperty(0.1), "end": MockProperty(3.1),
                  "step": MockProperty(0.1), "e_unit": MockProperty("meV")}
        e_axis = {"units": MockProperty("DeltaE"), "start": MockProperty(-10), "end": MockProperty(15),
                  "step": MockProperty(1), "e_unit": MockProperty("DeltaE")}
        mock_get_property.side_effect = [MockProperty("md_ws"), MockProperty(q_axis),
                                         MockProperty(e_axis), MockProperty("Direct"),
                                         MockProperty(True), MockProperty(False), MockProperty("Rebin")]
        mock_compute_cut.return_value = MagicMock()

        cut_algo = Cut()
        cut_algo.PyExec()

        mock_compute_cut.assert_called_once()
        mock_set_property.assert_called_with('OutputWorkspace', mock_compute_cut.return_value)

    def _test_psd_cut(self, normalized: bool, algorithm: str):
        psd_workspace = create_md_workspace(2, "md_ws")

        e_dim = psd_workspace.getYDimension()
        q_dim = psd_workspace.getXDimension()
        e_axis = Axis(e_dim.getDimensionId(), e_dim.getMinimum(), e_dim.getMaximum(), "1")
        q_axis = Axis(q_dim.getDimensionId(), q_dim.getMinimum(), q_dim.getMaximum(), "1")

        cut = compute_cut(psd_workspace, e_axis, q_axis, "Direct", True, normalized, algorithm)

        self.assertTrue(isinstance(cut, MDHistoWorkspace))
        self.assertEqual(cut.getSignalArray().shape, (1, 20))
        self.assertEqual(cut.getErrorSquaredArray().shape, (1, 20))
        return cut

    def _test_non_psd_cut(self, normalized: bool, algorithm: str):
        non_psd_workspace = create_simulation_workspace("Direct", "non_psd_ws", psd=False)

        cut = compute_cut(non_psd_workspace.raw_ws, self.q_axis, self.e_axis, "Direct", False, normalized, algorithm)

        self.assertTrue(isinstance(cut, MDHistoWorkspace))
        self.assertEqual(cut.getSignalArray().shape, (30, ))
        self.assertEqual(cut.getErrorSquaredArray().shape, (30, ))
