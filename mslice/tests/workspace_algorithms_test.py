from __future__ import (absolute_import, division, print_function)

import unittest

from mslice.models.workspacemanager.workspace_algorithms import (process_limits, scale_workspaces,
                                                                 export_workspace_to_ads, is_pixel_workspace, get_comment)
from mslice.models.workspacemanager.workspace_provider import add_workspace
from mslice.util.mantid.mantid_algorithms import (AppendSpectra, CreateSampleWorkspace, CreateSimulationWorkspace,
                                                  CreateWorkspace, ConvertToMD, AddSampleLog)

from mantid.api import AnalysisDataService


class WorkspaceAlgorithmsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.direct_workspace = CreateSimulationWorkspace(OutputWorkspace='MAR_workspace', Instrument='MAR',
                                                         BinParams=[-10, 1, 10], UnitX='DeltaE')
        cls.direct_workspace.e_mode = "Direct"
        cls.direct_workspace.e_fixed = 1.1

        cls.indirect_workspace = CreateSimulationWorkspace(OutputWorkspace='OSIRIS_workspace', Instrument='OSIRIS',
                                                           BinParams=[-10, 1, 10], UnitX='DeltaE')
        cls.indirect_workspace.e_mode = "Indirect"
        cls.indirect_workspace.e_fixed = 1.1

        CreateWorkspace(DataX=cls.indirect_workspace.raw_ws.readX(0), DataY=cls.indirect_workspace.raw_ws.readY(0),
                        ParentWorkspace='OSIRIS_workspace', UnitX='DeltaE', OutputWorkspace="extra_spectra_ws")

        cls.extended_workspace = AppendSpectra(InputWorkspace1='OSIRIS_workspace', InputWorkspace2='extra_spectra_ws',
                                               OutputWorkspace='extended_workspace')
        cls.extended_workspace.e_mode = "Indirect"
        cls.extended_workspace.e_fixed = 1.1

        sim_workspace = CreateSimulationWorkspace(Instrument='MAR', BinParams=[-10, 1, 10],
                                                  UnitX='DeltaE', OutputWorkspace='simws')
        AddSampleLog(sim_workspace, LogName='Ei', LogText='3.', LogType='Number')
        cls.pixel_workspace = ConvertToMD(InputWorkspace=sim_workspace, OutputWorkspace="convert_ws", QDimensions='|Q|',
                                          dEAnalysisMode='Direct', MinValues='-10,0,0', MaxValues='10,6,500',
                                          SplitInto='50,50')

        cls.test_workspace = CreateSampleWorkspace(OutputWorkspace='test_workspace', NumBanks=1, BankPixelWidth=5,
                                                   XMin=0.1, XMax=3.1, BinWidth=0.1, XUnit='DeltaE')


    @classmethod
    def tearDownClass(cls) -> None:
        AnalysisDataService.clear()

    def test_process_limits_does_not_fail_for_direct_data(self):
        process_limits(self.direct_workspace)

    def test_process_limits_does_not_fail_for_indirect_data(self):
        process_limits(self.indirect_workspace)

    def test_process_limits_will_raise_a_runtime_error_for_a_workspace_where_not_every_histogram_has_a_detector(self):
        with self.assertRaises(RuntimeError):
            process_limits(self.extended_workspace)

    def test_scale_workspaces_without_parameters(self):
        self.assertRaises(ValueError, scale_workspaces, self.direct_workspace)

    def test_scale_workspace_with_rebose(self):
        current_len = len(AnalysisDataService)
        add_workspace(self.test_workspace, self.test_workspace.name)
        export_workspace_to_ads(self.test_workspace)
        self.assertEqual(len(AnalysisDataService), current_len + 1)
        scale_workspaces([self.test_workspace], from_temp=300.0, to_temp=5.0)

    def test_if_pixel_workspace(self):
        self.assertTrue(is_pixel_workspace, self.pixel_workspace)

    def test_get_comment(self):
        self.assertEqual(get_comment(self.test_workspace), "")
