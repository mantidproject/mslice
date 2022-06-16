import numpy as np
from unittest import TestCase

from mantid.api import AnalysisDataService

from mslice.models.axis import Axis
from mslice.models.cut.cut_algorithm import compute_cut
from mslice.tests.testhelpers.workspace_creator import create_simulation_workspace, create_md_workspace


class CutAlgorithmTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.e_axis = Axis("DeltaE", -10, 15, 1)
        cls.q_axis = Axis("|Q|", 0.1, 3.1, 0.1)

        cls.md_ws = create_md_workspace(2, "md_ws")
        cls.non_psd_ws = create_simulation_workspace("Direct", "non_psd_ws", psd=False)

    @classmethod
    def tearDownClass(cls) -> None:
        AnalysisDataService.clear()

    def test_that_compute_cut_returns_a_normalized_result_for_PSD_data(self):
        cut = compute_cut(self.md_ws, self.q_axis, self.e_axis, "Direct", True, False, "Rebin")
        self.assertTrue(np.all(cut.getSignalArray() <= 1))
        self.assertTrue(np.all(cut.getErrorSquaredArray() <= 1))

    def test_that_compute_cut_returns_a_normalized_result_for_non_PSD_data(self):
        cut = compute_cut(self.non_psd_ws.raw_ws, self.q_axis, self.e_axis, "Direct", False, False, "Rebin")
        self.assertTrue(np.all(cut.getSignalArray() <= 1))
        self.assertTrue(np.all(cut.getErrorSquaredArray() <= 1))

    def test_that_compute_cut_returns_the_expected_results_for_PSD_data(self):
        pass

    def test_that_compute_cut_returns_the_expected_results_for_non_PSD_data(self):
        pass
