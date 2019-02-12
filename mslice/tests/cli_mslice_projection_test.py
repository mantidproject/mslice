import unittest
import mock
import numpy as np
from mantid.simpleapi import (AddSampleLog, CreateSampleWorkspace)

import mslice.cli._mslice_commands as mc
from mslice.workspace import wrap_workspace
import mslice.plotting.pyplot as plt
from mslice.cli.plotfunctions import pcolormesh, errorbar


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

    @mock.patch('matplotlib.axes.Axes.errorbar')
    @mock.patch('mslice.cli.projection_functions.errorbar')
    def test_that_mslice_projection_errorbar_works_correctly(self, plot_cut, errorbar):
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='mslice')
        workspace = self.create_workspace('cut')
        cut = mc.Cut(workspace)

        ax.errorbar(cut)
        plot_cut.assert_called()

        ax.errorbar('not_workspace')
        errorbar.assert_called()

    @mock.patch('matplotlib.axes.Axes.pcolormesh')
    @mock.patch('mslice.cli.projection_functions.pcolormesh')
    def test_that_mslice_projection_pcolormesh_works_correctly(self, plot_slice, pcolormesh):
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='mslice')
        workspace = self.create_workspace('slice')
        slice = mc.Slice(workspace)

        ax.pcolormesh(slice)
        plot_slice.assert_called()

        ax.pcolormesh('not_workspace')
        pcolormesh.assert_called()

    def test_that_plot_cut_mslice_projection_works_correctly(self):
        fig = plt.gcf()
        fig.add_subplot = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')
        workspace = self.create_workspace('cut')
        cut = mc.Cut(workspace)

        return_value = errorbar(ax, cut)
        self.assertEqual(ax.lines, return_value)

    def test_that_plot_slice_mslice_projection_works_correctly(self):
        fig = plt.gcf()
        fig.add_subplot = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')
        workspace = self.create_workspace('slice')
        slice = mc.Slice(workspace)

        return_value = pcolormesh(ax, slice)
        self.assertEqual(ax.collections[0], return_value)
