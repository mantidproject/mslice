from mock import MagicMock, patch, PropertyMock, ANY
import unittest

from matplotlib import colors
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from mslice.plotting.plot_window.slice_plot import SlicePlot
from mslice.plotting.plot_window.overplot_interface import toggle_overplot_line
from mslice.util.intensity_correction import IntensityType


class SlicePlotTest(unittest.TestCase):
    def setUp(self):
        self.plot_figure = MagicMock()
        self.plot_figure.window = MagicMock()
        self.axes = MagicMock()
        canvas = MagicMock()
        canvas.figure.gca = MagicMock(return_value=self.axes)
        self.plot_figure.window.canvas = canvas
        self.slice_plotter = MagicMock()
        self.slice_plotter.get_cached_sample_temp.return_value = ("test_log", True)

        self.slice_plot = SlicePlot(self.plot_figure, self.slice_plotter, "workspace")
        self.slice_plot.manager.report_as_current_and_return_previous_status.return_value = (
            None,
            False,
        )
        self.line = [Line2D([], [])]
        self.axes.get_legend_handles_labels = MagicMock(
            return_value=(self.line, ["some_label"])
        )

    def test_that_is_changed_works_correctly(self):
        self.slice_plot.default_options = {}

        self.slice_plot.default_options["y_grid"] = False
        self.slice_plot.y_grid = False
        self.slice_plot.default_options["x_grid"] = False
        self.slice_plot.x_grid = True

        self.assertEqual(self.slice_plot.is_changed("y_grid"), False)
        self.assertEqual(self.slice_plot.is_changed("x_grid"), True)

    def test_change_logarithmic(self):
        image = MagicMock()
        norm = PropertyMock(return_value=colors.Normalize)
        type(image).norm = norm
        self.axes.collections.__getitem__ = MagicMock(return_value=image)
        self.slice_plot.change_axis_scale((0, 10), True)
        image.set_norm.assert_called_once()
        norm_set = image.set_norm.call_args_list[0][0][0]
        self.assertEqual(norm_set.vmin, 0.001)
        self.assertEqual(norm_set.vmax, 10)
        self.assertEqual(type(norm_set), colors.LogNorm)
        image.set_clim.assert_called_once_with((0.001, 10))

    def test_change_linear(self):
        image = MagicMock()
        norm = PropertyMock(return_value=colors.LogNorm)
        type(image).norm = norm
        self.axes.collections.__getitem__ = MagicMock(return_value=image)
        self.slice_plot.change_axis_scale((0, 15), False)
        image.set_norm.assert_called_once()
        norm_set = image.set_norm.call_args_list[0][0][0]
        self.assertEqual(norm_set.vmin, 0)
        self.assertEqual(norm_set.vmax, 15)
        self.assertEqual(type(norm_set), colors.Normalize)
        image.set_clim.assert_called_once_with((0, 15))

    @patch("mslice.plotting.plot_window.slice_plot.SlicePlot._handle_temperature_input")
    def test_lines_redrawn_on_intensity_change(self, mock_handle_temperature_input):
        self.slice_plot.save_default_options()
        toggle_overplot_line(
            self.slice_plot,
            self.slice_plot._slice_plotter_presenter,
            self.slice_plot.plot_window.action_helium,
            4,
            True,
            True,
        )
        colorbar_range = PropertyMock(return_value=(0, 10))
        type(self.slice_plot).colorbar_range = colorbar_range
        self.slice_plotter.show_dynamical_susceptibility.__name__ = (
            "show_dynamical_susceptibility"
        )
        self.slice_plotter.show_dynamical_susceptibility.side_effect = ValueError()
        self.slice_plot.show_intensity_plot(
            self.plot_figure.action_chi_qe,
            self.slice_plotter.show_dynamical_susceptibility,
            True,
        )
        self.assertTrue(self.slice_plot.plot_window.action_set_temp_log.enabled)
        self.assertTrue(self.slice_plot.plot_window.action_helium.checked)
        self.slice_plotter.add_overplot_line.assert_any_call("workspace", 4, True, "")
        mock_handle_temperature_input.assert_called_with("test_log", True, False)


    @patch("mslice.plotting.plot_window.slice_plot.QtWidgets.QInputDialog.getInt")
    def test_arbitrary_recoil_line(self, qt_get_int_mock):
        qt_get_int_mock.return_value = (5, True)
        self.slice_plotter.add_overplot_line.reset_mock()
        self.plot_figure.action_arbitrary_nuclei.isChecked = MagicMock(
            return_value=True
        )

        self.slice_plot.arbitrary_recoil_line()
        self.slice_plotter.add_overplot_line.assert_called_once_with(
            "workspace", 5, True, None, False, 0, IntensityType.SCATTERING_FUNCTION
        )

    @patch("mslice.plotting.plot_window.slice_plot.QtWidgets.QInputDialog.getInt")
    def test_arbitrary_recoil_line_cancelled(self, qt_get_int_mock):
        qt_get_int_mock.return_value = (5, False)
        self.slice_plotter.add_overplot_line.reset_mock()
        self.plot_figure.action_arbitrary_nuclei.isChecked = MagicMock(
            return_value=True
        )

        self.slice_plot.arbitrary_recoil_line()
        self.slice_plotter.add_overplot_line.assert_not_called()

    def test_update_legend_in_icut_mode(self):
        self.slice_plot._canvas.manager.plot_handler.icut = MagicMock()

        self.slice_plot.update_legend()

        self.axes.get_legend.assert_called_once()
        if hasattr(Legend, "set_draggable"):
            self.axes.get_legend().set_draggable.assert_called_once()
        else:
            self.axes.get_legend().draggable.assert_called_once()

        self.assertEqual(
            self.slice_plot._canvas.manager.plot_handler.icut.rect.ax, self.axes
        )

    def test_update_legend_in_slice_plot(self):
        with (
            patch.object(self.axes, "legend") as mock_add_legend,
            patch.object(self.axes, "get_legend") as mock_get_legend,
        ):
            mock_get_legend.return_value = MagicMock()

            self.slice_plot.update_legend()
            mock_add_legend.assert_called_with(
                self.line, ["some_label"], fontsize=ANY, loc="upper right"
            )

    def test_all_fonts_size(self):
        slice_plot_colorbar_label_size = PropertyMock()
        slice_plot_colorbar_range_font_size = PropertyMock()
        type(self.slice_plot).colorbar_label_size = slice_plot_colorbar_label_size
        type(
            self.slice_plot
        ).colorbar_range_font_size = slice_plot_colorbar_range_font_size

        fonts_config = {
            "title_size": 15,
            "x_range_font_size": 14,
            "y_range_font_size": 13,
            "x_label_size": 12,
            "y_label_size": 11,
            "colorbar_label_size": 10,
            "colorbar_range_font_size": 9,
        }

        self.slice_plot.all_fonts_size = fonts_config

        slice_plot_colorbar_range_font_size.assert_called_once_with(9)
        slice_plot_colorbar_label_size.assert_called_once_with(10)

        self.assertEqual(self.slice_plot.title_size, 15)
        self.assertEqual(self.slice_plot.x_range_font_size, 14)
        self.assertEqual(self.slice_plot.y_range_font_size, 13)
        self.assertEqual(self.slice_plot.x_label_size, 12)
        self.assertEqual(self.slice_plot.y_label_size, 11)

    def test_increase_all_fonts(self):
        mock_colorbar_label_size = PropertyMock(return_value=9)
        mock_colorbar_range_font_size = PropertyMock(return_value=10)
        type(self.slice_plot).colorbar_label_size = mock_colorbar_label_size
        type(self.slice_plot).colorbar_range_font_size = mock_colorbar_range_font_size

        fonts_config = {
            "title_size": 15,
            "x_range_font_size": 14,
            "y_range_font_size": 13,
            "x_label_size": 12,
            "y_label_size": 11,
        }
        self.slice_plot.all_fonts_size = fonts_config

        self.slice_plot.increase_all_fonts()

        self.assertEqual(self.slice_plot.title_size, 16)
        self.assertEqual(self.slice_plot.x_range_font_size, 15)
        self.assertEqual(self.slice_plot.y_range_font_size, 14)
        self.assertEqual(self.slice_plot.x_label_size, 13)
        self.assertEqual(self.slice_plot.y_label_size, 12)
        mock_colorbar_label_size.assert_called_with(10)
        mock_colorbar_range_font_size.assert_called_with(11)


if __name__ == "__main__":
    unittest.main()
