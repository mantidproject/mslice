import unittest
import mock
import numpy as np
from mslice.models.axis import Axis
from mslice.models.cut.cut import Cut
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.util.intensity_correction import IntensityType
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
        cut = Cut(
            axis,
            integration_axis,
            intensity_start=3.0,
            intensity_end=11.0,
            norm_to_one=True,
            width=None,
        )
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

    def _workspace_handle_side_effect(self, workspace_name):
        key = list(self.cut_plotter_presenter._cut_cache_dict.keys())[0]
        for cut in self.cut_plotter_presenter._cut_cache_dict[key]:
            if cut.workspace_name == workspace_name:
                return cut.cut_ws

    @mock.patch(
        "mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.apply_intensity_correction_after_plot_over"
    )
    @mock.patch(
        "mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.get_current_plot_intensity"
    )
    @mock.patch("mslice.presenters.cut_plotter_presenter.get_workspace_handle")
    @mock.patch("mslice.presenters.cut_plotter_presenter.compute_cut")
    @mock.patch("mslice.presenters.cut_plotter_presenter.plot_cut_impl")
    def test_plot_single_cut_success(
        self,
        plot_cut_impl_mock,
        compute_cut_mock,
        get_ws_handle_mock,
        get_current_plot_intensity_mock,
        apply_intensity_mock,
    ):
        mock_ws = mock.MagicMock()
        mock_ws.name = "workspace"
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()
        cut_cache._update_cut_axis = mock.MagicMock()
        compute_cut_mock.return_value.axes = [
            cut_cache.cut_axis,
            cut_cache.integration_axis,
        ]
        get_current_plot_intensity_mock.return_value = False

        self.cut_plotter_presenter.run_cut("workspace", cut_cache)
        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(1, plot_cut_impl_mock.call_count)
        self.assertEqual(1, cut_cache._update_cut_axis.call_count)
        self.assertEqual(0, apply_intensity_mock.call_count)

    @mock.patch(
        "mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.update_main_window"
    )
    @mock.patch(
        "mslice.presenters.cut_plotter_presenter.CutPlotterPresenter.get_current_plot_intensity"
    )
    @mock.patch("mslice.models.cut.cut.compute_symmetrised")
    @mock.patch("mslice.models.cut.cut.compute_d2sigma")
    @mock.patch("mslice.models.cut.cut.compute_chi")
    @mock.patch("mslice.presenters.cut_plotter_presenter.compute_cut")
    @mock.patch("mslice.presenters.cut_plotter_presenter.plot_cut_impl")
    def test_plot_cut_with_intensity(
        self,
        plot_cut_impl_mock,
        compute_cut_mock,
        compute_chi_mock,
        compute_d2sigma_mock,
        compute_symmetrised_mock,
        get_current_plot_intensity_mock,
        update_main_window_mock,
    ):
        mock_ws = mock.MagicMock()
        mock_ws.name = "workspace"
        mock_ws.parent = "workspace_parent"
        cut_cache = self.create_cut_cache()
        cut_cache._update_cut_axis = mock.MagicMock()
        cut_cache._sample_temp = 100
        mock_intensity_ws = mock.MagicMock()
        mock_intensity_ws.get_signal.return_value = [0, 100]
        compute_cut_mock.return_value.axes = [
            cut_cache.cut_axis,
            cut_cache.integration_axis,
        ]
        compute_chi_mock.return_value = mock_intensity_ws
        compute_d2sigma_mock.return_value = mock_intensity_ws
        compute_symmetrised_mock.return_value = mock_intensity_ws
        self.main_presenter.is_energy_conversion_allowed.return_value = False
        intensity_correction_types = [
            IntensityType.CHI,
            IntensityType.CHI_MAGNETIC,
            IntensityType.D2SIGMA,
            IntensityType.SYMMETRISED,
        ]

        for intensity in intensity_correction_types:
            get_current_plot_intensity_mock.return_value = intensity
            self.cut_plotter_presenter._plot_cut(
                mock_ws, cut_cache, False, intensity_correction=intensity
            )
            compute_cut_mock.assert_called_with(
                mock_ws,
                cut_cache.cut_axis,
                cut_cache.integration_axis,
                cut_cache.norm_to_one,
                cut_cache.algorithm,
                True,
            )
            cut_cache._cut_ws = None

        self.assertEqual(
            1 * len(intensity_correction_types), plot_cut_impl_mock.call_count
        )
        self.assertEqual(
            1 * len(intensity_correction_types), cut_cache._update_cut_axis.call_count
        )
        self.assertEqual(
            1 * len(intensity_correction_types),
            get_current_plot_intensity_mock.call_count,
        )
        self.assertEqual(
            1 * len(intensity_correction_types), update_main_window_mock.call_count
        )
        self.assertEqual(2, compute_chi_mock.call_count)

    @mock.patch(
        "mslice.presenters.cut_plotter_presenter.Axis.validate_step_against_workspace"
    )
    @mock.patch("mslice.presenters.cut_plotter_presenter.get_workspace_handle")
    @mock.patch("mslice.presenters.cut_plotter_presenter.CutPlotterPresenter._plot_cut")
    def test_multiple_cuts_with_width(
        self, plot_cut_mock, get_ws_handle_mock, validate_step_mock
    ):
        mock_ws = mock.MagicMock()
        mock_ws.name = "workspace"
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()
        cut_cache.width = "25"

        self.cut_plotter_presenter.run_cut("workspace", cut_cache)
        self.assertEqual(4, plot_cut_mock.call_count)
        # 0.0 now because the integration_axis is reset when a cut is made to allow
        # the script generator use the same cut intervals
        self.assertEqual(0.0, cut_cache.integration_axis.start)
        self.assertEqual(None, cut_cache.cut_ws)
        validate_step_mock.assert_called_once_with(mock_ws)

    @mock.patch("mslice.presenters.cut_plotter_presenter.get_workspace_handle")
    @mock.patch("mslice.presenters.cut_plotter_presenter.compute_cut")
    @mock.patch("mslice.presenters.cut_plotter_presenter.plot_cut_impl")
    @mock.patch("mslice.presenters.cut_plotter_presenter.export_workspace_to_ads")
    def test_save_to_workspace_success(
        self,
        export_workspace_to_ads,
        plot_cut_impl_mock,
        compute_cut_mock,
        get_ws_handle_mock,
    ):
        mock_ws = mock.MagicMock()
        mock_ws.name = "workspace"
        get_ws_handle_mock.return_value = mock_ws
        cut_cache = self.create_cut_cache()

        self.cut_plotter_presenter.run_cut("workspace", cut_cache, save_only=True)
        self.assertEqual(1, compute_cut_mock.call_count)
        self.assertEqual(1, export_workspace_to_ads.call_count)
        self.assertEqual(0, plot_cut_impl_mock.call_count)

    @mock.patch("mslice.presenters.cut_plotter_presenter.get_workspace_handle")
    @mock.patch("mslice.presenters.cut_plotter_presenter.compute_cut")
    @mock.patch("mslice.presenters.cut_plotter_presenter.plot_cut_impl")
    def test_plot_cut_from_workspace(
        self, plot_cut_impl_mock, compute_cut_mock, get_ws_handle_mock
    ):
        mock_ws = mock.MagicMock()
        mock_ws.name = "workspace"
        get_ws_handle_mock.return_value = mock_ws
        self.main_presenter.get_selected_workspaces.return_value = ["workspace"]

        self.cut_plotter_presenter.plot_cut_from_selected_workspace(plot_over=True)
        self.assertEqual(0, compute_cut_mock.call_count)
        self.assertEqual(1, plot_cut_impl_mock.call_count)

    @mock.patch("mslice.presenters.cut_plotter_presenter.get_workspace_handle")
    @mock.patch("mslice.presenters.cut_plotter_presenter.CutPlotterPresenter._plot_cut")
    @mock.patch("mslice.presenters.cut_plotter_presenter.draw_interactive_cut")
    def test_plot_interactive_cut(
        self, draw_interact_mock, plot_cut_mock, get_ws_handle_mock
    ):
        mock_ws = mock.MagicMock()
        mock_ws.name = "workspace"
        get_ws_handle_mock.return_value = mock_ws
        cut_axis = Axis("units", "0", "100", "1")
        integration_axis = Axis("units", 0.0, 100.0, 0)
        cut = Cut(cut_axis, integration_axis, None, None)
        self.cut_plotter_presenter.plot_interactive_cut("workspace", cut, False, False)

        self.assertEqual(1, plot_cut_mock.call_count)
        self.assertEqual(1, draw_interact_mock.call_count)

    @mock.patch("mslice.presenters.cut_plotter_presenter.cut_figure_exists")
    def test_set_is_icut(self, cut_figure_exists):
        cut_figure_exists.return_value = True

        self.cut_plotter_presenter.set_is_icut(False)

    def test_store_and_get_icut(self):
        return_value = self.cut_plotter_presenter.get_icut()
        self.assertEqual(return_value, None)

        self.cut_plotter_presenter.store_icut("icut")
        return_value = self.cut_plotter_presenter.get_icut()
        self.assertEqual(return_value, "icut")

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
        self.cut_plotter_presenter._temp_cut_cache = list(
            self.cut_plotter_presenter._cut_cache_dict[ax]
        )
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

    @mock.patch("mslice.presenters.cut_plotter_presenter.plt.gca")
    def test_missing_sample_temperature(self):
        ax = mock.MagicMock()
        self.populate_presenter_cache_dict(ax)
        cut_1, cut_2, cut_3 = self.cut_plotter_presenter._cut_cache_dict[ax]
        cut_1.sample_temp = 120
        mock_plot_gca.return_value = ax
        self.cut_plotter_presenter._get_overall_max_signal(
            IntensityType.SCATTERING_FUNCTION
        )
        self.assertEqual(cut_1.sample_temp, 120)
        self.assertEqual(cut_2.sample_temp, 120)

    @mock.patch("mslice.presenters.cut_plotter_presenter.plt.gca")
    def test_get_overall_q_axis(self, mock_plot_gca):
        ax = mock.MagicMock()
        self.populate_presenter_cache_dict(ax)
        cut_1, cut_2, cut_3 = self.cut_plotter_presenter._cut_cache_dict[ax]
        cut_1._integration_axis = Axis("|Q|", -10.0, 10.0, 0)
        cut_2._integration_axis = Axis("|Q|", 0.0, 20.0, 0)
        cut_3._integration_axis = Axis("|Q|", 10.0, 30.0, 0)
        mock_plot_gca.return_value = ax

        overall_axis = self.cut_plotter_presenter._get_overall_q_axis()
        self.assertEqual(overall_axis, Axis("|Q|", -10.0, 30.0, 0))

    @mock.patch("mslice.presenters.cut_plotter_presenter.plt.gca")
    def test_get_overall_max_signal(self, mock_plot_gca):
        ax = mock.MagicMock()
        self.populate_presenter_cache_dict(ax)
        cut_1, cut_2, cut_3 = self.cut_plotter_presenter._cut_cache_dict[ax]
        cut_1._cut_ws.get_signal.return_value = 25.0
        cut_2._cut_ws.get_signal.return_value = 50.0
        cut_3._cut_ws.get_signal.return_value = 100.0
        mock_plot_gca.return_value = ax

        max_signal = self.cut_plotter_presenter._get_overall_max_signal(
            IntensityType.SCATTERING_FUNCTION
        )
        self.assertEqual(max_signal, 100)

    @mock.patch("mslice.presenters.cut_plotter_presenter.compute_powder_line")
    @mock.patch("mslice.presenters.cut_plotter_presenter.plot_overplot_line")
    def test_add_overplot_line_will_use_ten_percent_of_max_signal(
        self, mock_plot_overplot_line, mock_compute_powder_list
    ):
        cache = mock.MagicMock()
        cache.rotated = False
        self.cut_plotter_presenter._cut_cache_dict = mock.MagicMock()
        self.cut_plotter_presenter._cut_cache_dict.__getitem__.return_value = [cache]

        self.cut_plotter_presenter._get_overall_max_signal = mock.MagicMock()
        self.cut_plotter_presenter._get_overall_max_signal.return_value = 4.0

        self.cut_plotter_presenter._get_overall_q_axis = mock.MagicMock()
        self.cut_plotter_presenter._get_overall_q_axis.return_value = Axis(
            "A-1", "1.0", "2.0", "0.1"
        )

        x, y = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        key, recoil = "Copper", False
        mock_compute_powder_list.return_value = (x, y)

        self.cut_plotter_presenter.add_overplot_line("test_ws (-5.5,5.5)", key, recoil)

        args = mock_plot_overplot_line.call_args.args
        np.testing.assert_allclose(args[0], x)
        np.testing.assert_allclose(args[1], y / 10)

    @mock.patch("mslice.presenters.cut_plotter_presenter.get_workspace_handle")
    @mock.patch("mslice.presenters.cut_plotter_presenter.CutPlotterPresenter._plot_cut")
    def test_show_intensity(self, mock_plot_cut, mock_get_workspace_handle):
        ax = mock.MagicMock()
        self.populate_presenter_cache_dict(ax)
        cut_cache = self.cut_plotter_presenter._cut_cache_dict[ax]
        cut_1, cut_2, cut_3 = cut_cache
        mock_get_workspace_handle.side_effect = self._workspace_handle_side_effect
        self.cut_plotter_presenter._show_intensity(
            cut_cache, "dynamical_susceptibility"
        )
        cut_1_call = mock.call(
            cut_1._cut_ws,
            cut_1,
            plot_over=False,
            intensity_correction="dynamical_susceptibility",
        )
        cut_2_call = mock.call(
            cut_2._cut_ws,
            cut_2,
            plot_over=True,
            intensity_correction="dynamical_susceptibility",
        )
        cut_3_call = mock.call(
            cut_3._cut_ws,
            cut_3,
            plot_over=True,
            intensity_correction="dynamical_susceptibility",
        )
        mock_plot_cut.assert_has_calls([cut_1_call, cut_2_call, cut_3_call])

    @mock.patch("mslice.presenters.cut_plotter_presenter.CutPlotterPresenter._plot_cut")
    def test_plot_with_width_returns_a_warning_string_from_validate_function(
        self, mock_plot_cut
    ):
        mock_ws = mock.MagicMock()
        mock_cut = mock.MagicMock()
        mock_cut.cut_axis.validate_step_against_workspace.return_value = (
            "step size warning"
        )
        mock_cut.integration_axis.start = 0.2
        mock_cut.integration_axis.end = 0.3
        mock_cut.width = 0.01

        self.assertEqual(
            "step size warning",
            self.cut_plotter_presenter._plot_with_width(mock_ws, mock_cut, False),
        )
        mock_plot_cut.assert_called()
