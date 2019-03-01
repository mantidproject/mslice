import unittest
import mock
import numpy as np
from mslice.workspace import wrap_workspace
from mantid.simpleapi import AddSampleLog, CreateSampleWorkspace


class ScriptingTest(unittest.TestCase):

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

    @mock.patch('mslice.scripting.preprocess_lines')
    @mock.patch('mslice.scripting.add_plot_statements')
    @mock.patch('mslice.cli._mslice_commands.GlobalFigureManager')
    @mock.patch('mslice.scripting.generate_script')
    def test_that_generate_script_works_as_expected(self, gs, gfm, add_plot_statements, preprocess_lines):
        workspace = self.create_workspace('test')
        filename = 'filename'
        plot_handler = gfm.get_active_figure().plot_handler
        plot_window = plot_handler.plot_window

        gs(workspace.name, filename=filename, plot_handler=plot_handler, window=plot_window)

        add_plot_statements.assert_called_once()
        preprocess_lines.assert_called_once()
