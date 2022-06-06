from __future__ import (absolute_import, division, print_function)

import unittest

from mslice.models.workspacemanager.workspace_algorithms import process_limits
from mslice.util.mantid.mantid_algorithms import AppendSpectra, CreateSimulationWorkspace, CreateWorkspace

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
