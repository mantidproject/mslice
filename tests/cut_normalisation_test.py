import numpy as np
from unittest import TestCase
import warnings

from mantid.api import AnalysisDataService

from mslice.models.cut.cut_normalisation import normalize_workspace
from tests.testhelpers.workspace_creator import create_md_histo_workspace, create_workspace


class CutNormalisationTest(TestCase):

    def setUp(self):
        self.non_md_histo_ws = create_workspace("non_md_histo_ws")
        self.md_histo_ws = create_md_histo_workspace(2, "md_histo_ws")

    def tearDown(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            AnalysisDataService.clear()

    def test_that_normalize_workspace_fails_for_a_non_IMDHistoWorkspace(self):
        try:
            normalize_workspace(self.non_md_histo_ws)
            self.fail("The normalise_workspace function did not fail as expected.")
        except AssertionError:
            pass

    def test_that_normalize_workspace_sets_the_expected_comment(self):
        normalize_workspace(self.md_histo_ws)
        self.assertEqual(self.md_histo_ws.getComment(), "Normalized By MSlice")

    def test_that_normalize_workspace_will_normalize_the_data_as_expected(self):
        normalize_workspace(self.md_histo_ws)
        self.assertTrue(np.all(self.md_histo_ws.getSignalArray() <= 1))
        self.assertTrue(np.all(self.md_histo_ws.getErrorSquaredArray() <= 1))
