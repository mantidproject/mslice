import mslice.util.mantid.init_mantid # noqa: F401

import unittest
import mock
import numpy as np
from mantid.simpleapi import (AddSampleLog, CreateSampleWorkspace, CreateMDHistoWorkspace, CreateSimulationWorkspace,
                              ConvertToMD)

import mslice.cli as mc
from mslice.cli.cli_mslice_projection_functions import PlotCutMsliceProjection, PlotSliceMsliceProjection
import matplotlib.pyplot as plt
from mslice.workspace import wrap_workspace


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
        workspace.limits['DeltaE'] = [-10,15,1]
        workspace.e_fixed = 10
        return workspace

    def create_pixel_workspace(self, name):
        sim_workspace = CreateSimulationWorkspace(Instrument='MAR', BinParams=[-10, 1, 10],
                                                  UnitX='DeltaE', OutputWorkspace=name)
        AddSampleLog(sim_workspace, LogName='Ei', LogText='3.', LogType='Number')
        sim_workspace = ConvertToMD(InputWorkspace=sim_workspace, OutputWorkspace=name, QDimensions='|Q|',
                                    dEAnalysisMode='Direct', MinValues='-10,0,0', MaxValues='10,6,500',
                                    SplitInto='50,50')
        workspace = wrap_workspace(sim_workspace, name)
        workspace.is_PSD = True
        workspace.limits['MomentumTransfer'] = [0.1, 3.1, 0.1]
        workspace.limits['|Q|'] = [0.1, 3.1, 0.1]
        workspace.limits['DeltaE'] = [-10, 15, 1]
        workspace.e_fixed = 10
        workspace.ef_defined = True
        return workspace

    def create_histo_workspace(self, name):
        signal = list(range(0, 100))
        error = np.zeros(100) + 2
        workspace = CreateMDHistoWorkspace(Dimensionality=2, Extents='0,100,0,100',
                                           SignalInput=signal, ErrorInput=error,
                                           NumberOfBins='10,10', Names='Dim1,Dim2',
                                           Units='U,U', OutputWorkspace=name)
        workspace = wrap_workspace(workspace, name)
        workspace.is_PSD = True
        return workspace

    @mock.patch('mslice.cli.cli_mslice_projection_functions.cli_cut_plotter_presenter')
    def test_that_mslice_projection_plot_cut_works_correctly(self, cut_presenter):
        cut_presenter.plot_cut_from_workspace = mock.MagicMock()
        fig, ax = plt.subplots(subplot_kw={'projection': 'mslice'})
        workspace = self.create_workspace('test_plot_cut_non_psd_cli')
        cut = mc.Cut(workspace)

        ax.plot(cut)

        intensity_range = None
        PlotOver = False
        is_gui = False
        cut_presenter.plot_cut_from_workspace.assert_called_once_with(cut, intensity_range=intensity_range,
                                                                      plot_over=PlotOver, is_gui=is_gui)


    @mock.patch('mslice.cli.cli_mslice_projection_functions.cli_slice_plotter_presenter')
    def test_that_mslice_projection_slice_cut_works_correctly(self, slice_presenter):
        slice_presenter.plot_from_cache = mock.MagicMock()
        fig, ax = plt.subplots(subplot_kw={'projection': 'mslice'})
        workspace = self.create_workspace('test_plot_cut_non_psd_cli')
        slice_ws = mc.Slice(workspace)

        ax.pcolormesh(slice_ws)

        slice_presenter.plot_from_cache.assert_called_once_with(slice_ws)
