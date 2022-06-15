import numpy as np
from unittest import TestCase

from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateMDHistoWorkspace

from mslice.models.cut.cut_normalisation import normalize_workspace
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace


class CutNormalisationTest(TestCase):

    def setUp(self):
        self.non_md_histo_ws = CreateSampleWorkspace(OutputWorkspace="non_md_histo_ws", NumBanks=1, BankPixelWidth=5,
                                                     XMin=0.1, XMax=3.1, BinWidth=0.1, XUnit='DeltaE')

        self.md_histo_ws = CreateMDHistoWorkspace(Dimensionality=2, Extents="-3,3,-10,10",
                                                  SignalInput=list(range(0, 100)), ErrorInput=list(range(0, 100)),
                                                  NumberOfBins="10,10", Names="Dim1,Dim2",
                                                  Units="MomentumTransfer,EnergyTransfer",
                                                  OutputWorkspace="md_histo_ws")

    def tearDown(self) -> None:
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
