import numpy as np
from unittest import TestCase

from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateMDWorkspace, FakeMDEventData, BinMD


class CutAlgorithmTest(TestCase):

    def tearDown(self) -> None:
        AnalysisDataService.clear()

    def test_that_compute_cut_returns_a_result_with_the_expected_size_for_normalized_psd_rebin_data(self):
        AnalysisDataService.clear()
        psd_workspace = CreateMDWorkspace(Dimensions=2, Extents=",".join(["-10,10"] * 2),
                                          Names=",".join(["|Q|", "DeltaE"]),
                                          Units=",".join(["U"] * 2), OutputWorkspace="cut_algo_test_md_ws")
        FakeMDEventData(InputWorkspace=psd_workspace, PeakParams="500000,0,0,3",
                        RandomizeSignal=False)
        e_dim = psd_workspace.getYDimension()
        q_dim = psd_workspace.getXDimension()

        y_dim = e_dim.getDimensionId() + "," + str(e_dim.getMinimum()) + "," + str(e_dim.getMaximum()) + "," + "20"
        x_dim = q_dim.getDimensionId() + "," + str(q_dim.getMinimum()) + "," + str(q_dim.getMaximum()) + "," + "1"

        cut = BinMD(InputWorkspace=psd_workspace, AxisAligned=True, AlignedDim0=x_dim, AlignedDim1=y_dim)
        self.assertAlmostEqual(np.nanmax(cut.getSignalArray()), 101381, 3)

        self.assertEqual(cut.getXDimension().getBinWidth(), 20.0)
        self.assertEqual(cut.getYDimension().getBinWidth(), 1.0)
