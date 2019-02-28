from mock import MagicMock, patch, ANY
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
        xy_config = {'x_log': False, 'y_log': False, 'x_range': (0, 10), 'y_range': (1, 7)}

        self.cut_plot.change_axis_scale(xy_config)
        self.axes.set_xscale.assert_called_once_with('linear')
        self.axes.set_yscale.assert_called_once_with('linear')
        self.assertEqual(self.cut_plot.x_range, (0, 10))
        self.assertEqual(self.cut_plot.y_range, (1, 7))

    def test_change_scale_log(self):
        self.cut_plot.save_default_options()
        self.axes.set_xscale = MagicMock()
        self.axes.set_yscale = MagicMock()
        line = MagicMock()
        line.get_ydata = MagicMock(return_value=np.array([1, 5, 10]))
        line.get_xdata = MagicMock(return_value=np.array([20, 60, 12]))
        self.axes.get_lines = MagicMock(return_value=[line])
        self.canvas.figure.gca = MagicMock(return_value=self.axes)
        xy_config = {'x_log': True, 'y_log': True, 'x_range': (0, 20), 'y_range': (1, 7)}

        self.cut_plot.change_axis_scale(xy_config)
        self.axes.set_xscale.assert_called_once_with('symlog', linthreshx=10.0)
        self.axes.set_yscale.assert_called_once_with('symlog', linthreshy=1.0)
        self.assertEqual(self.cut_plot.x_range, (0, 20))
        self.assertEqual(self.cut_plot.y_range, (1, 7))

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

    def test_update_legend(self):
        line = Line2D([], [])
        self.axes.get_legend_handles_labels = MagicMock(return_value=([line], ['some_label']))
        self.cut_plot.update_legend()
        self.assertTrue(self.cut_plot._legends_visible[0])
        self.axes.legend.assert_called_with([line], ['some_label'], fontsize=ANY)
