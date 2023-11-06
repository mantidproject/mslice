from __future__ import (absolute_import, division, print_function)

import unittest

from mslice.models.axis import Axis
from mslice.models.workspacemanager.workspace_algorithms import (process_limits, process_limits_event, scale_workspaces,
                                                                 export_workspace_to_ads, is_pixel_workspace,
                                                                 get_axis_from_dimension, get_comment, remove_workspace_from_ads)
from mslice.models.workspacemanager.workspace_provider import add_workspace
from tests.testhelpers.workspace_creator import (create_md_histo_workspace, create_workspace,
                                                 create_simulation_workspace)
from mslice.util.mantid.mantid_algorithms import (AppendSpectra, CloneWorkspace, CreateSimulationWorkspace,
                                                  CreateWorkspace, ConvertToMD, AddSampleLog)
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mantid.api import AnalysisDataService


class WorkspaceAlgorithmsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.direct_workspace = create_simulation_workspace("Direct", "MAR_workspace")
        cls.indirect_workspace = create_simulation_workspace("Indirect", "OSIRIS_workspace")

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
        cls.pixel_workspace.limits['|Q|'] = [0.1, 3.1, 0.1]
        cls.pixel_workspace.limits['DeltaE'] = [-10, 15, 1]

        cls.test_workspace = create_workspace('test_workspace')
        cls.histo_workspace = HistogramWorkspace(create_md_histo_workspace(2, 'histo_workspace'), 'histo_workspace')

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

    def test_process_limits_events_will_raise_an_attribute_error_for_pixel_workspace(self):
        with self.assertRaises(AttributeError):
            process_limits_event(self.pixel_workspace)

    def test_get_axis_from_dimension(self):
        return_value = get_axis_from_dimension(self.pixel_workspace, 0)
        self.assertEqual(return_value, Axis('|Q|', '0.1', '3.1', '0.1'))
        return_value = get_axis_from_dimension(self.pixel_workspace, 1)
        self.assertEqual(return_value, Axis('DeltaE', '-10.0', '15.0', '1.0'))

    def test_scale_workspaces_without_parameters(self):
        self.assertRaises(ValueError, scale_workspaces, self.direct_workspace)

    def test_scale_workspace_with_rebose(self):
        current_len = len(AnalysisDataService)
        export_workspace_to_ads(self.test_workspace)
        self.assertEqual(len(AnalysisDataService), current_len + 1)
        scale_workspaces([self.test_workspace], from_temp=300.0, to_temp=5.0)

    def test_if_pixel_workspace(self):
        self.assertTrue(is_pixel_workspace, self.pixel_workspace)

    def test_export_workspace_to_ads_does_not_fail_for_histo_workspace(self):
        add_workspace(self.histo_workspace, self.histo_workspace.name)
        export_workspace_to_ads(self.histo_workspace)

    def test_get_comment(self):
        self.assertEqual(get_comment(self.test_workspace), "")

    def test_remove_workspace_from_ads(self):
        test_workspace2 = CloneWorkspace(OutputWorkspace='test_workspace2', InputWorkspace=self.test_workspace)
        export_workspace_to_ads(test_workspace2)
        print(AnalysisDataService.getObjectNames())
        print("~~~~~~~~~~")
        current_len = len(AnalysisDataService)
        remove_workspace_from_ads(test_workspace2)
        print(AnalysisDataService.getObjectNames())
        self.assertEqual(len(AnalysisDataService), current_len - 1)
