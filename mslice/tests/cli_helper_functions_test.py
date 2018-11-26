import unittest
import mock
import numpy as np
from mantid.simpleapi import (AddSampleLog, CreateSampleWorkspace, CreateMDHistoWorkspace, CreateSimulationWorkspace,
                              ConvertToMD)
from mslice.workspace import wrap_workspace
from mslice.cli.cli_helperfunctions import _string_to_axis, _string_to_integration_axis, _process_axis,\
    _check_workspace_name, _check_workspace_type, is_gui, is_slice, is_cut
from mslice.cli import Cut, Slice
from mslice.models.axis import Axis
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.workspace.workspace import Workspace as MatrixWorkspace
from mslice.presenters.busy import show_busy
from mslice.cli.cli_helper_classes.cli_data_loader import CLIDataLoaderWidget

class CLIHelperFunctionsTest(unittest.TestCase):

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

    def test_that_string_to_axis_works_as_expected(self):
        string = "name,1,5,0.1"
        return_value = _string_to_axis(string)
        self.assertEqual(return_value, Axis('name', '1', '5', '.1'))

    def test_that_string_to_integration_axis_works_as_expected(self):
        string = "name,1,5"
        return_value = _string_to_integration_axis(string)
        self.assertEqual(return_value, Axis('name', '1', '5', 4))

        string2 = "name,1,5,0.1"
        return_value = _string_to_integration_axis(string2)
        self.assertEqual(return_value, Axis('name', '1', '5', 0.1))

    def test_that_process_axis_works_as_expected(self):
        axis = 'DeltaE'
        workspace = self.create_workspace("test_workspace")
        fallback_index = 0

        return_value = _process_axis(axis, fallback_index, workspace)

        self.assertEqual(return_value, Axis(units='DeltaE', start=-10, end=15, step=1))

    def test_that_check_workspace_name_works_as_expected(self):
        workspace = self.create_workspace('test_workspace')

        return_value = _check_workspace_name(workspace)
        self.assertEqual(return_value, None)

        with self.assertRaises(TypeError):
            _check_workspace_name(3)

        with self.assertRaises(TypeError):
            _check_workspace_name("not_a_workspace")

    def test_that_check_workspace_type_works_as_expected(self):
        psd_workspace = self.create_pixel_workspace('psd_workspace')
        workspace = self.create_histo_workspace('histogram_workspace')

        return_value = _check_workspace_type(workspace, HistogramWorkspace)
        self.assertEqual(return_value, None)

        with self.assertRaises(RuntimeError):
            _check_workspace_type(psd_workspace, MatrixWorkspace)

    def test_that_is_slice_works_as_expected(self):
        workspace = self.create_workspace('workspace')
        slice_ws = Slice(workspace)
        hist_ws = self.create_histo_workspace('hist_workspace')

        return_value = is_slice(slice_ws)
        self.assertEqual(return_value, True)

        with self.assertRaises(ValueError):
            is_slice(hist_ws)

    def test_that_is_cut_works_as_expected(self):
        workspace = self.create_workspace('workspace')
        cut_ws = Cut(workspace)
        slice_ws = Slice(workspace)

        return_value = is_cut(cut_ws)
        self.assertEqual(return_value, True)

        with self.assertRaises(ValueError):
            is_cut(slice_ws)

    def test_that_is_gui_works_as_expected(self):
        return_value = is_gui()
        self.assertEqual(return_value, False)

    def test_that_cli_data_loader_widget_view_is_used_correctly_with_busy(self):
        data_loader_widget = CLIDataLoaderWidget()
        return_value = show_busy(data_loader_widget)
        self.assertNotEquals(return_value, None)
