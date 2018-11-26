import unittest
import mock
from mslice.models.axis import Axis
from mslice.models.cut.cut import Cut
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface


class CutPlotterPresenterTest(unittest.TestCase):

    def setUp(self):
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.cut_plotter_presenter = CutPlotterPresenter()
        self.cut_plotter_presenter.set_is_icut = mock.MagicMock()
        self.cut_plotter_presenter.register_master(self.main_presenter)

    def create_cut_cache(self):
        axis = Axis("units", "0", "100", "1")
        integration_axis = Axis("units", 0.0, 100.0, 0)
        return Cut(axis, integration_axis, intensity_start=3.0, intensity_end= 11.0, norm_to_one=True, width=None)

    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    def test_plot_single_cut_success(self, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()

        self.cut_plotter_presenter.run_cut('workspace', cut_cache)
        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(1, plot_cut_impl_mock.call_count)

    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    def test_multiple_cuts_with_width(self, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()
        cut_cache.width = '25'

        self.cut_plotter_presenter.run_cut('workspace', cut_cache)
        self.assertEqual(4, compute_cut_mock.call_count)
        self.assertEqual(4, plot_cut_impl_mock.call_count)
        self.assertEqual(75, cut_cache.integration_axis.start)

    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    def test_save_to_workspace_success(self, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()

        self.cut_plotter_presenter.run_cut('workspace', cut_cache, save_only=True)
        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(0, plot_cut_impl_mock.call_count)

    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    def test_plot_cut_from_workspace(self, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        self.main_presenter.get_selected_workspaces.return_value = ['workspace']

        self.cut_plotter_presenter.plot_cut_from_selected_workspace(plot_over=True)
        self.assertEqual(0, compute_cut_mock.call_count)
        self.assertEqual(1, plot_cut_impl_mock.call_count)

    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    @mock.patch('mslice.presenters.cut_plotter_presenter.draw_interactive_cut')
    def test_plot_interactive_cut(self, draw_interact_mock, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_axis = Axis("units", "0", "100", "1")
        integration_axis = Axis("units", 0.0, 100.0, 0)
        self.cut_plotter_presenter.plot_interactive_cut('workspace', cut_axis, integration_axis, False)

        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(1, plot_cut_impl_mock.call_count)
        self.assertEqual(1, draw_interact_mock.call_count)
