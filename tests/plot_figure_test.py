import unittest
from unittest import mock
from unittest.mock import patch

from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.util.intensity_correction import IntensityType

FORCE_METHOD_CALLS_TO_QAPP_THREAD = 'mslice.plotting.plot_window.plot_figure_manager.force_method_calls_to_qapp_thread'


class PlotFigureTest(unittest.TestCase):

    def setUp(self):
        self.mock_force_qapp = mock.patch(
            FORCE_METHOD_CALLS_TO_QAPP_THREAD).start()
        # make it a noop
        self.mock_force_qapp.side_effect = lambda arg: arg
        from mslice.plotting.plot_window.plot_figure_manager import new_plot_figure_manager
        self.new_plot_figure_manager = new_plot_figure_manager
        self.slice_plotter_presenter = SlicePlotterPresenter()
        self.cut_plotter_presenter = CutPlotterPresenter()

    def tearDown(self):
        self.mock_force_qapp.stop()

    def test_save_slice_nexus_sofqe(self):
        gman = mock.Mock()
        workspace = 'testworkspace'
        file_name = ('', 'test.nxs', '.nxs')
        fg = self.new_plot_figure_manager(num=1, global_manager=gman)
        fg.add_slice_plot(self.slice_plotter_presenter, workspace=workspace)

        with patch('mslice.plotting.plot_window.plot_figure_manager.get_save_directory') as get_save_dir, \
             patch('mslice.models.workspacemanager.workspace_algorithms.save_nexus') as save_nexus, \
             patch('mslice.models.workspacemanager.workspace_algorithms.get_workspace_handle') as get_handle, \
             patch.object(SlicePlotterPresenter, 'get_slice_cache') as get_slice_cache:
            get_save_dir.return_value = file_name
            slice_cache = mock.Mock()
            slice_cache.scattering_function = workspace
            get_slice_cache.return_value = slice_cache
            get_handle.return_value = workspace
            fg.save_plot()
            save_nexus.assert_called_once_with(workspace, file_name[1])
            get_slice_cache.assert_called_once()

    def test_save_slice_matlab_gdos(self):
        gman = mock.Mock()
        workspace = 'testworkspace'
        file_name = ('', 'test.mat', '.mat')
        fg = self.new_plot_figure_manager(num=1, global_manager=gman)
        fg.add_slice_plot(self.slice_plotter_presenter, workspace=workspace)

        with patch('mslice.plotting.plot_window.plot_figure_manager.get_save_directory') as get_save_dir, \
             patch('mslice.models.workspacemanager.workspace_algorithms.save_matlab') as save_matlab, \
             patch('mslice.models.workspacemanager.workspace_algorithms.get_workspace_handle') as get_handle, \
             patch.object(SlicePlotterPresenter, 'get_slice_cache') as get_slice_cache:
            get_save_dir.return_value = file_name
            slice_cache = mock.Mock()
            slice_cache.gdos = workspace
            get_slice_cache.return_value = slice_cache
            get_handle.return_value = workspace
            fg.plot_handler.intensity_type = IntensityType.GDOS
            fg.save_plot()
            save_matlab.assert_called_once_with(workspace, file_name[1])
            get_slice_cache.assert_called_once()
