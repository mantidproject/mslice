import mslice.util.mantid.init_mantid # noqa: F401

import unittest
import mock
import numpy as np
from mantid.simpleapi import (AddSampleLog, CreateSampleWorkspace, CreateMDHistoWorkspace, CreateSimulationWorkspace,
                              ConvertToMD)

from mslice.cli._mslice_commands import Load, MakeProjection, Slice, Cut
from mslice.models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator
from mslice.presenters.powder_projection_presenter import PowderProjectionPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.views.interfaces.powder_projection_view import PowderView
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.workspace import wrap_workspace
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.workspace import Workspace


class CommandLineTest(unittest.TestCase):

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


    @mock.patch('mslice.cli._mslice_commands.get_workspace_handle')
    @mock.patch('mslice.cli._mslice_commands.app')
    @mock.patch('mslice.cli._mslice_commands.ospath.exists')
    def test_load(self, ospath_mock, app_mock, get_workspace_mock):
        app_mock.MAIN_WINDOW = mock.Mock()
        ospath_mock.exists.return_value=True
        path = 'made_up_path'
        Load(path)
        app_mock.MAIN_WINDOW.dataloader_presenter.load_workspace.assert_called_once_with([path])
        get_workspace_mock.assert_called_with(path)

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_projection(self,  app_mock):
        app_mock.MAIN_WINDOW.powder_presenter = PowderProjectionPresenter(mock.create_autospec(PowderView), MantidProjectionCalculator())
        app_mock.MAIN_WINDOW.powder_presenter.register_master(mock.create_autospec(MainPresenterInterface))
        workspace = self.create_workspace('test_projection_cli')
        workspace.is_PSD = True
        result = MakeProjection(workspace, '|Q|', 'DeltaE')
        signal = result.get_signal()
        self.assertEqual(type(result), PixelWorkspace)
        self.assertAlmostEqual(signal[0][0], 0, 4)
        self.assertAlmostEqual(signal[0][11], 0.1753, 4)
        self.assertAlmostEqual(signal[0][28], 0.4248, 4)

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_projection_fail_non_PSD(self, app_mock):
        app_mock.MAIN_WINDOW.powder_presenter = PowderProjectionPresenter(mock.create_autospec(PowderView),
                                                                          MantidProjectionCalculator())
        workspace = self.create_workspace('test_projection_cli')
        with self.assertRaises(RuntimeError):
            MakeProjection(workspace, '|Q|', 'DeltaE')


    @mock.patch('mslice.cli._mslice_commands.app')
    def test_slice_non_psd(self, app_mock):
        app_mock.MAIN_WINDOW.slice_plotter_presenter = SlicePlotterPresenter()
        workspace = self.create_workspace('test_slice_cli')
        result = Slice(workspace)
        signal = result.get_signal()
        self.assertEqual(type(result), Workspace)
        self.assertAlmostEqual(0.4250, signal[1][10], 4)
        self.assertAlmostEqual(0.9250, signal[4][11], 4)

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_slice_non_psd_axes_specified(self, app_mock):
        app_mock.MAIN_WINDOW.slice_plotter_presenter = SlicePlotterPresenter()
        workspace = self.create_workspace('test_slice_cli_axes')
        result = Slice(workspace, 'DeltaE,0,15,1', '|Q|,0,3,0.1')
        signal = result.get_signal()
        self.assertEqual(type(result), Workspace)
        self.assertAlmostEqual(0.4250, signal[2][0], 4)
        self.assertAlmostEqual(0.9250, signal[5][1], 4)

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_cut_non_psd(self, app_mock):
        app_mock.MAIN_WINDOW.cut_plotter_presenter = CutPlotterPresenter()
        app_mock.MAIN_WINDOW.cut_plotter_presenter.register_master(mock.create_autospec(MainPresenterInterface))
        workspace = self.create_workspace('test_workspace_cut_cli')
        result = Cut(workspace)
        signal = result.get_signal()
        self.assertEqual(type(result), HistogramWorkspace)
        self.assertAlmostEqual(1.1299, signal[5], 4)
        self.assertAlmostEqual(1.375, signal[8], 4)

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_slice_psd(self, app_mock):
        app_mock.MAIN_WINDOW.slice_plotter_presenter = SlicePlotterPresenter()
        workspace = self.create_pixel_workspace('test_slice_psd_cli')
        result = Slice(workspace)
        signal = result.get_signal()
        self.assertEqual(type(result), HistogramWorkspace)
        self.assertEqual(60, signal[0][9])
        self.assertEqual(110, signal[3][7])
        self.assertEqual(105, signal[5][6])

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_cut_psd(self, app_mock):
        app_mock.MAIN_WINDOW.cut_plotter_presenter = CutPlotterPresenter()
        app_mock.MAIN_WINDOW.cut_plotter_presenter.register_master(mock.create_autospec(MainPresenterInterface))
        workspace = self.create_pixel_workspace('test_workspace_cut_psd_cli')
        result = Cut(workspace)
        signal = result.get_signal()
        self.assertEqual(type(result), HistogramWorkspace)
        self.assertEqual(128, signal[0])
        self.assertEqual(192, signal[29])
        self.assertEqual(429, signal[15])

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_slice_fail_workspace_type(self, app_mock):
        app_mock.MAIN_WINDOW.slice_plotter_presenter = SlicePlotterPresenter()
        workspace = self.create_histo_workspace('test_slice_fail_type_cli')
        with self.assertRaises(RuntimeError):
            Slice(workspace)

    @mock.patch('mslice.cli._mslice_commands.app')
    def test_cut_fail_workspace_type(self, app_mock):
        app_mock.MAIN_WINDOW.cut_plotter_presenter = CutPlotterPresenter()
        app_mock.MAIN_WINDOW.cut_plotter_presenter.register_master(mock.create_autospec(MainPresenterInterface))
        workspace = self.create_histo_workspace('test_cut_fail_type_cli')
        with self.assertRaises(RuntimeError):
            Cut(workspace)
