import unittest
import mock
import numpy as np
from mantid.simpleapi import (AddSampleLog, CreateSampleWorkspace)

import mslice.cli._mslice_commands as mc
from mslice.workspace import wrap_workspace
from mslice.plotting.globalfiguremanager import GlobalFigureManager
from mslice.plotting.plot_window.cut_plot import CutPlot
from mslice.cli.plotfunctions import pcolormesh, errorbar
from mslice.cli.helperfunctions import _get_overplot_key


class CLIProjectionTest(unittest.TestCase):

    def setUp(self):

        self.ax = None
        self.cut = None
        self.slice = None
        self.workspace = None

        fig = GlobalFigureManager.get_active_figure().figure
        self.ax = fig.add_subplot(111, projection='mslice')
        self.workspace = self.create_workspace('workspace')
        self.cut = mc.Cut(self.workspace)
        self.slice = mc.Slice(self.workspace)

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
    @mock.patch('mslice.cli.plotfunctions.errorbar')
    @mock.patch('mslice.cli._mslice_commands.is_gui')
    def test_that_mslice_projection_errorbar_works_correctly(self, is_gui, plot_cut, errorbar):
        is_gui.return_value = True

        self.ax.errorbar(self.cut)
        plot_cut.assert_called()

        self.ax.errorbar('not_workspace')
        errorbar.assert_called()

    @mock.patch('matplotlib.axes.Axes.pcolormesh')
    @mock.patch('mslice.cli.plotfunctions.pcolormesh')
    def test_that_mslice_projection_pcolormesh_works_correctly(self, plot_slice, pcolormesh):
        self.ax.pcolormesh(self.slice)
        plot_slice.assert_called()

        self.ax.pcolormesh('not_workspace')
        pcolormesh.assert_called()

    @mock.patch('mslice.cli._mslice_commands.is_gui')
    def test_that_plot_cut_mslice_projection_works_correctly(self, is_gui):
        is_gui.return_value = True
        cut = mc.Cut(self.workspace)

        fig = mock.MagicMock()
        ax = fig.add_subplot(111, projection='mslice')
        ax.get_ylim.return_value = (0., 1.)

        return_value = errorbar(ax, cut)
        self.assertEqual(ax.lines, return_value)

    def test_that_plot_slice_mslice_projection_works_correctly(self):
        slice = mc.Slice(self.workspace)

        return_value = pcolormesh(self.ax, slice)
        self.assertEqual(self.ax.collections[0], return_value)

    @mock.patch('mslice.cli._update_overplot_checklist')
    @mock.patch('mslice.cli._update_legend')
    @mock.patch('mslice.app.presenters.get_slice_plotter_presenter')
    def test_that_recoil_works_as_expected_without_relative_molecular_mass(self, get_spp, update_legend, update_check):
        element = 'Hydrogen'
        key = _get_overplot_key(element, rmm=None)

        self.ax.recoil(self.workspace, element, rmm=None)

        get_spp().add_overplot_line.assert_called_with(self.workspace.name, key, recoil=True, cif=None)
        update_legend.assert_called_once_with()
        update_check.assert_called_with(key)

    @mock.patch('mslice.cli.GlobalFigureManager')
    @mock.patch('mslice.cli._update_overplot_checklist')
    @mock.patch('mslice.cli._update_legend')
    @mock.patch('mslice.app.presenters.get_slice_plotter_presenter')
    def test_that_recoil_works_as_expected_with_relative_molecular_mass(self, spp, update_legend, update_check, gfm):
        rmm = 34
        key = _get_overplot_key(element=None, rmm=rmm)
        plot_handler = gfm.get_active_figure().plot_handler

        self.ax.recoil(self.workspace, rmm=rmm)

        spp().add_overplot_line.assert_called_with(self.workspace.name, key, recoil=True, cif=None)
        update_legend.assert_called_once_with()
        update_check.assert_called_with(key)
        plot_handler._arb_nuclei_rmm = rmm

    @mock.patch('mslice.cli._update_overplot_checklist')
    @mock.patch('mslice.cli._update_legend')
    @mock.patch('mslice.app.presenters.get_slice_plotter_presenter')
    def test_that_bragg_works_as_expected_without_cif(self, get_spp, update_legend, update_check):
        element = 'Tantalum'
        key = _get_overplot_key(element, rmm=None)

        self.ax.bragg(self.workspace, element)

        get_spp().add_overplot_line.assert_called_with(self.workspace.name, key, recoil=False, cif=None)
        update_legend.assert_called_once_with()
        update_check.assert_called_with(key)

    @mock.patch('mslice.cli.is_gui')
    @mock.patch('mslice.cli.GlobalFigureManager')
    @mock.patch('matplotlib.axes.Axes.grid')
    def test_that_grid_works_as_expected(self, grid, gfm, is_gui):
        plot_handler = gfm.get_active_figure().plot_handler
        is_gui.return_value = False
        b = True
        which = 'major'

        self.ax.grid(b=b, which=which, axis='x')
        grid.assert_called_with(self.ax, b, which, 'x')
        self.assertEqual(plot_handler.manager._xgrid, b)

        self.ax.grid(b=b, which=which, axis='y')
        grid.assert_called_with(self.ax, b, which, 'y')
        self.assertEqual(plot_handler.manager._xgrid, b)

    def test_that_waterfall_command_works(self):
        active_figure = mock.MagicMock()
        active_figure.plot_handler = mock.MagicMock(spec=CutPlot)

        fig = GlobalFigureManager.get_active_figure().figure
        ax = fig.add_subplot(111, projection='mslice')
        with mock.patch('mslice.cli._mslice_commands.GlobalFigureManager.get_active_figure') as gaf:
            gaf.return_value = active_figure
            ax.set_waterfall(True, 1, 2)

        self.assertEqual(active_figure.plot_handler.waterfall, True)
        self.assertEqual(active_figure.plot_handler.waterfall_x, 1)
        self.assertEqual(active_figure.plot_handler.waterfall_y, 2)
        active_figure.plot_handler.toggle_waterfall.assert_called()
