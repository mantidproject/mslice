import numpy as np

from mock import patch
from unittest import TestCase

from mantid.api import AlgorithmFactory, AnalysisDataService
from mantid.simpleapi import AddSampleLog, CreateMDWorkspace, _create_algorithm_function

from mslice.models.axis import Axis
from mslice.models.cut.cut_algorithm import Cut
from mslice.models.cut.cut_functions import compute_cut, is_cuttable, output_workspace_name
from mslice.util.mantid.algorithm_wrapper import wrap_algorithm
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace
from mslice.workspace.pixel_workspace import PixelWorkspace


class SliceFunctionsTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace_name = "TestWorkspaceName"
        cls.integration_start = 0
        cls.integration_end = 1.5

        cls.sim_scattering_data = np.arange(cls.integration_start, cls.integration_end, 0.002).reshape(30, 25).transpose()
        cls.e_axis = Axis('DeltaE', -10, 15, 1)
        cls.q_axis = Axis('|Q|', 0.1, 3.1, 0.1)
        cls.q_axis_degrees = Axis('Degrees', 3, 33, 1)

        cls.test_ws = CreateSampleWorkspace(OutputWorkspace=cls.workspace_name, NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                            XMax=3.1, BinWidth=0.1, XUnit='DeltaE')

        for i in range(cls.test_ws.raw_ws.getNumberHistograms()):
            cls.test_ws.raw_ws.setY(i, cls.sim_scattering_data[i])
        AddSampleLog(workspace=cls.test_ws.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)
        cls.test_ws.e_mode = 'Direct'
        cls.test_ws.e_fixed = 3
        cls.test_ws.is_PSD = False

        md_2d_workspace = CreateMDWorkspace(Dimensions=2, Extents='-10,10,-10,10', Names='A,B', Units='U,U',
                                            OutputWorkspace="2d_workspace")
        cls.pixel_workspace = PixelWorkspace(md_2d_workspace, '2d_workspace')

        md_3d_workspace = CreateMDWorkspace(Dimensions=3, Extents='-10,10,-10,10,-10,10', Names='A,B,C', Units='U,U,U',
                                            OutputWorkspace="3d_workspace")
        cls.workspace_3d = PixelWorkspace(md_3d_workspace, "3d_workspace")

    @classmethod
    def tearDownClass(cls) -> None:
        AnalysisDataService.clear()

    def test_that_output_workspace_name_returns_the_expected_result(self):
        self.assertEqual(output_workspace_name(self.workspace_name, self.integration_start, self.integration_end),
                         "TestWorkspaceName_cut(0.000,1.500)")

    def test_that_output_workspace_name_will_round_the_integration_start_and_end_to_three_decimals(self):
        self.assertEqual(output_workspace_name(self.workspace_name, 1.23456, 5.67891),
                         "TestWorkspaceName_cut(1.235,5.679)")

    @patch('mslice.models.cut.cut_functions.mantid_algorithms')
    def test_compute_cut_returns_the_expected_result(self, alg_mock):
        AlgorithmFactory.subscribe(Cut)
        alg_mock.Cut = wrap_algorithm(_create_algorithm_function('Cut', 1, Cut()))

        cut = compute_cut(self.test_ws, self.q_axis, self.e_axis, False)

        self.assertEqual(cut.parent, self.workspace_name)
        self.assertEqual(cut.get_signal().shape, (30,))

    def test_that_is_cuttable_returns_true_for_a_2D_pixel_workspace(self):
        self.assertTrue(is_cuttable(self.pixel_workspace))

    def test_that_is_cuttable_returns_false_for_a_non_2d_workspace(self):
        self.assertFalse(is_cuttable(self.workspace_3d))

    def test_that_is_cuttable_returns_true_for_a_valid_workspace2D(self):
        self.assertTrue(is_cuttable(self.test_ws))
