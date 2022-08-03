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

    def populate_presenter_cache_dict(self, ax):
        cut_1 = self.create_cut_cache()
        cut_ws_1 = mock.MagicMock()
        cut_ws_1.name = "ws_1"
        cut_1._cut_ws = cut_ws_1
        cut_1.parent_ws_name = "parent_ws_1"

        cut_2 = self.create_cut_cache()
        cut_ws_2 = mock.MagicMock()
        cut_ws_2.name = "ws_2"
        cut_2._cut_ws = cut_ws_2
        cut_2.parent_ws_name = "parent_ws_1"

        cut_3 = self.create_cut_cache()
        cut_ws_3 = mock.MagicMock()
        cut_ws_3.name = "ws_3"
        cut_3._cut_ws = cut_ws_3
        cut_3.parent_ws_name = "parent_ws_2"

        self.cut_plotter_presenter._cut_cache_dict[ax] = [cut_1, cut_2, cut_3]

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

    def test_save_cache_plot_over(self):
        cut_ws = mock.MagicMock()
        cut_ws.intensity_corrected = False
        cut_1 = self.create_cut_cache()
        cut_2 = self.create_cut_cache()
        cut_1._cut_ws = cut_ws
        cut_2._cut_ws = cut_ws
        ax = mock.MagicMock()
        self.cut_plotter_presenter.save_cache(ax, cut_1, True)
        self.cut_plotter_presenter.save_cache(ax, cut_2, True)
        self.assertEqual(len(self.cut_plotter_presenter._cut_cache_dict[ax]), 2)
        self.assertEqual(self.cut_plotter_presenter._cut_cache_dict[ax][0], cut_1)
        self.assertEqual(self.cut_plotter_presenter._cut_cache_dict[ax][1], cut_2)

    def test_save_cache_no_plot_over(self):
        cut_ws = mock.MagicMock()
        cut_ws.intensity_corrected = False
        cut_1 = self.create_cut_cache()
        cut_2 = self.create_cut_cache()
        cut_1._cut_ws = cut_ws
        cut_2._cut_ws = cut_ws
        ax = mock.MagicMock()
        self.cut_plotter_presenter.save_cache(ax, cut_1, False)
        self.cut_plotter_presenter.save_cache(ax, cut_2, False)
        self.assertEqual(len(self.cut_plotter_presenter._cut_cache_dict[ax]), 1)
        self.assertEqual(self.cut_plotter_presenter._cut_cache_dict[ax][0], cut_2)

    def test_save_cache_with_intensity(self):
        cut_ws_1 = mock.MagicMock()
        cut_ws_1.intensity_corrected = False
        cut_ws_2 = mock.MagicMock()
        cut_ws_2.intensity_corrected = True
        cut_1 = self.create_cut_cache()
        cut_2 = self.create_cut_cache()
        cut_1._cut_ws = cut_ws_1
        cut_2._cut_ws = cut_ws_2
        ax = mock.MagicMock()
        self.cut_plotter_presenter.save_cache(ax, cut_1, True)
        self.cut_plotter_presenter.save_cache(ax, cut_2, True)
        self.assertEqual(len(self.cut_plotter_presenter._cut_cache_dict[ax]), 1)
        self.assertEqual(self.cut_plotter_presenter._cut_cache_dict[ax][0], cut_1)

    def test_save_cache_duplicate_no_plot_over(self):
        cut_ws = mock.MagicMock()
        cut_ws.intensity_corrected = False
        cut_1 = self.create_cut_cache()
        cut_2 = self.create_cut_cache()
        cut_1._cut_ws = cut_ws
        cut_2._cut_ws = cut_ws
        ax = mock.MagicMock()
        self.cut_plotter_presenter.save_cache(ax, cut_1, True)
        self.cut_plotter_presenter._temp_cut_cache = list(self.cut_plotter_presenter._cut_cache_dict[ax])
        self.cut_plotter_presenter.save_cache(ax, cut_2, False)
        self.assertEqual(len(self.cut_plotter_presenter._cut_cache_dict[ax]), 1)
        self.assertEqual(self.cut_plotter_presenter._cut_cache_dict[ax][0], cut_1)

    def test_set_sample_temp_parent_in_cache(self):
        ax = mock.MagicMock()
        self.populate_presenter_cache_dict(ax)
        cut_1, cut_2, cut_3 = self.cut_plotter_presenter._cut_cache_dict[ax]

        self.cut_plotter_presenter.set_sample_temperature(ax, "ws_1", 100)
        self.assertEqual(cut_1.sample_temp, 100)
        self.assertEqual(cut_2.sample_temp, 100)
        self.assertEqual(cut_3._sample_temp, None)

    def test_propagate_sample_temperatures_throughout_cache(self):
        ax = mock.MagicMock()
        self.populate_presenter_cache_dict(ax)
        cut_1, cut_2, cut_3 = self.cut_plotter_presenter._cut_cache_dict[ax]
        cut_1.sample_temp = 120
        self.cut_plotter_presenter.propagate_sample_temperatures_throughout_cache(ax)
        self.assertEqual(cut_1.sample_temp, 120)
        self.assertEqual(cut_2.sample_temp, 120)
        self.assertEqual(cut_3._sample_temp, None)
