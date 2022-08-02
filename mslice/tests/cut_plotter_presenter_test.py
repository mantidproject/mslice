import unittest
import mock
from mslice.models.axis import Axis
from mslice.models.cut.cut import Cut
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
import mslice.plotting.globalfiguremanager as gfm


class CutPlotterPresenterTest(unittest.TestCase):

    def setUp(self):
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.cut_plotter_presenter = CutPlotterPresenter()
        self.cut_plotter_presenter.register_master(self.main_presenter)
        gfm.GlobalFigureManager.activate_category(gfm.CATEGORY_CUT)

    def create_cut_cache(self):
        axis = Axis("units", "0", "100", "1")
        integration_axis = Axis("units", 0.0, 100.0, 0)
        cut = Cut(axis, integration_axis, intensity_start=3.0, intensity_end=11.0, norm_to_one=True, width=None)
        return cut

    @mock.patch('mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.apply_intensity_correction_after_plot_over')
    @mock.patch('mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.get_current_plot_intensity')
    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    def test_plot_single_cut_success(self, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock,
                                     get_current_plot_intensity_mock, apply_intensity_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()
        cut_cache._update_cut_axis = mock.MagicMock()
        compute_cut_mock.return_value.axes = [cut_cache.cut_axis, cut_cache.integration_axis]
        get_current_plot_intensity_mock.return_value = False

        self.cut_plotter_presenter.run_cut('workspace', cut_cache)
        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(1, plot_cut_impl_mock.call_count)
        self.assertEqual(1, cut_cache._update_cut_axis.call_count)
        self.assertEqual(0, apply_intensity_mock.call_count)

    @mock.patch('mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.apply_intensity_correction_after_plot_over')
    @mock.patch('mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.get_current_plot_intensity')
    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    def test_plot_single_cut_success_with_intensity(self, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock,
                                                    get_current_plot_intensity_mock, apply_intensity_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()
        cut_cache._update_cut_axis = mock.MagicMock()
        compute_cut_mock.return_value.axes = [cut_cache.cut_axis, cut_cache.integration_axis]
        get_current_plot_intensity_mock.return_value = "show_dynamic_susceptibility"

        self.cut_plotter_presenter.run_cut('workspace', cut_cache, True)
        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(1, plot_cut_impl_mock.call_count)
        self.assertEqual(1, cut_cache._update_cut_axis.call_count)
        self.assertEqual(1, apply_intensity_mock.call_count)

    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.CutPlotterPresenter._plot_cut')
    def test_multiple_cuts_with_width(self, plot_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()
        cut_cache.width = '25'

        self.cut_plotter_presenter.run_cut('workspace', cut_cache)
        self.assertEqual(4, plot_cut_mock.call_count)
        # 0.0 now because the integration_axis is reset when a cut is made to allow
        # the script generator use the same cut intervals
        self.assertEqual(0.0, cut_cache.integration_axis.start)
        self.assertEqual(None, cut_cache.cut_ws)

    @mock.patch('mslice.presenters.cut_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.cut_plotter_presenter.compute_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.plot_cut_impl')
    @mock.patch('mslice.presenters.cut_plotter_presenter.export_workspace_to_ads')
    def test_save_to_workspace_success(self, export_workspace_to_ads, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()

        self.cut_plotter_presenter.run_cut('workspace', cut_cache, save_only=True)
        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(1, export_workspace_to_ads.call_count)
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
    @mock.patch('mslice.presenters.cut_plotter_presenter.CutPlotterPresenter._plot_cut')
    @mock.patch('mslice.presenters.cut_plotter_presenter.draw_interactive_cut')
    def test_plot_interactive_cut(self, draw_interact_mock, plot_cut_mock, get_ws_handle_mock):
        mock_ws = mock.MagicMock()
        mock_ws.name = 'workspace'
        get_ws_handle_mock.return_value = mock_ws
        cut_axis = Axis("units", "0", "100", "1")
        integration_axis = Axis("units", 0.0, 100.0, 0)
        cut = Cut(cut_axis, integration_axis, None, None)
        self.cut_plotter_presenter.plot_interactive_cut('workspace', cut, False, False)

        self.assertEqual(1, plot_cut_mock.call_count)
        self.assertEqual(1, draw_interact_mock.call_count)

    @mock.patch('mslice.presenters.cut_plotter_presenter.cut_figure_exists')
    def test_set_is_icut(self, cut_figure_exists):
        cut_figure_exists.return_value = True

        self.cut_plotter_presenter.set_is_icut(False)

    def test_store_and_get_icut(self):
        return_value = self.cut_plotter_presenter.get_icut()
        self.assertEquals(return_value, None)

        self.cut_plotter_presenter.store_icut('icut')
        return_value = self.cut_plotter_presenter.get_icut()
        self.assertEquals(return_value, 'icut')

    def test_workspace_selection_changed(self):
        self.cut_plotter_presenter.workspace_selection_changed()
