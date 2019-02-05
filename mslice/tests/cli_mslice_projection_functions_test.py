import unittest
import mock
import numpy as np
from mantid.simpleapi import (AddSampleLog, CreateSampleWorkspace)

import mslice.cli._mslice_commands as mc
from mslice.workspace import wrap_workspace
import mslice.plotting.pyplot as plt


class CLIProjectionTest(unittest.TestCase):

    def create_workspace(self, name):
        workspace = CreateSampleWorkspace(OutputWorkspace=name, NumBanks=1, BankPixelWidth=5, XMin=0.1,
                                          XMax=3.1, BinWidth=0.1, XUnit='DeltaE')
        AddSampleLog(Workspace=workspace, LogName='Ei', LogText='3.', LogType='Number')
        sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25).transpose()
        for i in range(workspace.getNumberHistograms()):
            workspace.setY(i, sim_scattering_data[i])
        workspace = wrap_workspace(workspace, name)
        workspace.is_PSD = False
        workspace.limits['MomentumTransfer'] = [0.1, 3.1, 0.1]
        workspace.limits['|Q|'] = [0.1, 3.1, 0.1]
        workspace.limits['DeltaE'] = [-10, 15, 1]
        workspace.e_fixed = 10
        return workspace

    @mock.patch('mslice.cli.projection_functions.plotfunctions')
    def test_that_mslice_projection_plot_cut_works_correctly(self, plotfunctions):
        fig = plt.gcf()
        fig.add_subplot = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')
        workspace = self.create_workspace('test_plot_cut_non_psd_cli')
        cut = mc.Cut(workspace)

        ax.errorbar(cut)

        plotfunctions.errorbar.assert_called()

    @mock.patch('mslice.cli.projection_functions.plotfunctions')
    def test_that_mslice_projection_slice_cut_works_correctly(self, plotfunctions):
        fig = plt.gcf()
        fig.add_subplot = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')
        workspace = self.create_workspace('test_plot_cut_non_psd_cli')
        slice_ws = mc.Slice(workspace)

        ax.pcolormesh(slice_ws)

        plotfunctions.pcolormesh.assert_called()
