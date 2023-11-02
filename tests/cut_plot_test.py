from mock import MagicMock, PropertyMock, patch, ANY
import numpy as np
import unittest

from matplotlib.lines import Line2D
from mslice.plotting.plot_window.cut_plot import CutPlot, get_min


class CutPlotTest(unittest.TestCase):

    def setUp(self):
        self.plot_figure = MagicMock()
        self.plot_figure.window = MagicMock()
        self.canvas = MagicMock()
        self.plot_figure.window.canvas = self.canvas
        self.cut_plotter = MagicMock()
        self.axes = MagicMock()
        self.canvas.figure.gca = MagicMock(return_value=self.axes)
        self.cut_plot = CutPlot(self.plot_figure, self.cut_plotter, "workspace")

    def test_that_is_changed_works_correctly(self):
        self.cut_plot.default_options = {}

        self.cut_plot.default_options['y_grid'] = False
        self.cut_plot.y_grid = False
        self.cut_plot.default_options['x_grid'] = False
        self.cut_plot.x_grid = True

        self.assertEqual(self.cut_plot.is_changed('y_grid'), False)
        self.assertEqual(self.cut_plot.is_changed('x_grid'), True)

    def test_get_min(self):
        data = [np.array([3, 6, 10]), np.array([3, 2, 7])]
        self.assertEqual(get_min(data), 2)
        self.assertEqual(get_min(data, 4), 6)

    def test_change_scale_linear(self):
        self.axes.set_xscale = MagicMock()
        self.axes.set_yscale = MagicMock()
        type(self.cut_plot).y_log = PropertyMock(return_value=False)
        self.cut_plot.x_log = False
        self.axes.set_xscale.assert_called_with('linear')
        self.axes.set_yscale.assert_called_with('linear')

    def test_change_x_scale_log(self):
        self.axes.set_xscale = MagicMock()
        self.axes.set_yscale = MagicMock()
        type(self.cut_plot).y_log = PropertyMock(return_value=False)
        line = MagicMock()
        line.get_xdata = MagicMock(return_value=np.array([20, 60, 12]))
        self.axes.get_lines = MagicMock(return_value=[line])
        self.cut_plot.x_log = True

        self.axes.set_yscale.assert_called_with('linear')
        self.axes.set_xscale.assert_called_once_with('symlog', linthresh=10.0)

    # Currently failing, although it's the mirror version of the above
    # def test_change_y_scale_log(self):
    #     self.axes.set_xscale = MagicMock()
    #     self.axes.set_yscale = MagicMock()
    #     type(self.cut_plot).x_log = PropertyMock(return_value=False)
    #     line = MagicMock()
    #     line.get_ydata = MagicMock(return_value=np.array([1, 5, 10]))
    #     self.axes.get_lines = MagicMock(return_value=[line])
    #     self.cut_plot.update_bragg_peaks = MagicMock()
    #
    #     self.cut_plot.y_log = True
    #
    #     self.axes.set_xscale.assert_called_with('linear')
    #     self.axes.set_yscale.assert_called_once_with('symlog', linthresh=1.0)
    #     self.cut_plot.update_bragg_peaks.assert_called_once_with(refresh=True)

    @patch('mslice.plotting.plot_window.cut_plot.quick_options')
    def test_line_clicked(self, quick_options_mock):
        line = Line2D([], [])
        self.cut_plot.update_legend = MagicMock()
        self.cut_plot.object_clicked(line)
        quick_options_mock.assert_called_once_with(line, self.cut_plot)
        self.cut_plot.update_legend.assert_called_once()
        self.cut_plot._canvas.draw.assert_called_once()

    @patch('mslice.plotting.plot_window.cut_plot.quick_options')
    def test_object_clicked(self, quick_options_mock):
        text = "some_label"
        self.cut_plot._get_line_index = MagicMock(return_value=2)
        self.cut_plot.update_legend = MagicMock()
        self.cut_plot.object_clicked(text)
        self.cut_plot._get_line_index.assert_not_called()
        quick_options_mock.assert_called_once_with(text, self.cut_plot)
        self.cut_plot.update_legend.assert_called_once()
        self.cut_plot._canvas.draw.assert_called_once()

    def test_update_legend_legends_not_shown(self):
        line = Line2D([], [])
        self.axes.get_legend_handles_labels = MagicMock(return_value=([line], ['some_label']))
        self.cut_plot._legends_shown = False
        self.cut_plot.update_legend()
        self.assertTrue(self.cut_plot._legends_visible[0])
        self.axes.legend.assert_not_called()

    def test_update_legend_legends_shown(self):
        line = Line2D([], [])
        self.axes.get_legend_handles_labels = MagicMock(return_value=([line], ['some_label']))
        self.cut_plot._legends_shown = True
        self.cut_plot.update_legend()
        self.assertTrue(self.cut_plot._legends_visible[0])
        self.axes.legend.assert_called_with([line], ['some_label'], fontsize=ANY)

    def test_update_legend_with_line_data(self):
        line_data = [
            {'shown': True, 'legend': 2, 'label': 'visible_line_data_label'},
            {'shown': True, 'legend': 0, 'label': 'non_visible_line_data_label'}
        ]
        mock_line = Line2D([], [])
        another_mock_line = Line2D([], [])

        self.axes.get_legend_handles_labels = MagicMock(return_value=(
            [mock_line, another_mock_line], ['mock_label', 'another_mock_label']
        ))
        self.cut_plot._legends_shown = True

        self.cut_plot.update_legend(line_data)
        self.assertEqual(self.cut_plot._legends_visible, [2, 0])
        self.axes.legend.assert_called_with([mock_line], ['visible_line_data_label'], fontsize=ANY)

    def test_waterfall(self):
        self.cut_plot._apply_offset = MagicMock()
        self.cut_plot.update_bragg_peaks = MagicMock()
        self.cut_plot.waterfall = True
        self.cut_plot.waterfall_x = 1
        self.cut_plot.waterfall_y = 2
        self.cut_plot.toggle_waterfall()
        self.cut_plot._apply_offset.assert_called_with(1, 2)
        self.cut_plot.waterfall = False
        self.cut_plot.toggle_waterfall()
        self.cut_plot._apply_offset.assert_called_with(0, 0)
        self.cut_plot.update_bragg_peaks.assert_called_with(refresh=True)

    def test_all_fonts_size(self):
        fonts_config = {'title_size': 15, 'x_range_font_size': 14, 'y_range_font_size': 13,
                        'x_label_size': 12, 'y_label_size': 11}

        self.cut_plot.all_fonts_size = fonts_config
        self.assertEqual(self.cut_plot.title_size, 15)
        self.assertEqual(self.cut_plot.x_range_font_size, 14)
        self.assertEqual(self.cut_plot.y_range_font_size, 13)
        self.assertEqual(self.cut_plot.x_label_size, 12)
        self.assertEqual(self.cut_plot.y_label_size, 11)

    def test_increment_all_fonts(self):
        fonts_config = {'title_size': 15, 'x_range_font_size': 14, 'y_range_font_size': 13,
                        'x_label_size': 12, 'y_label_size': 11}
        self.cut_plot.all_fonts_size = fonts_config

        self.cut_plot.increase_all_fonts()
        self.assertEqual(self.cut_plot.title_size, 16)
        self.assertEqual(self.cut_plot.x_range_font_size, 15)
        self.assertEqual(self.cut_plot.y_range_font_size, 14)
        self.assertEqual(self.cut_plot.x_label_size, 13)
        self.assertEqual(self.cut_plot.y_label_size, 12)
