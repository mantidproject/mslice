from mock import MagicMock
import numpy as np
import unittest


from mslice.plotting.plot_window.cut_plot import CutPlot


class CutPlotTest(unittest.TestCase):

    def setUp(self):
        self.plot_figure = MagicMock()
        self.canvas = MagicMock()
        self.cut_plotter = MagicMock()
        self.axes = MagicMock()
        self.canvas.figure.gca = MagicMock(return_value=self.axes)
        self.cut_plot = CutPlot(self.plot_figure, self.canvas, self.cut_plotter)

    def test_get_min(self):
        data = [np.array([3, 6, 10]), np.array([3, 2, 7])]
        self.assertEqual(self.cut_plot.get_min(data), 2)
        self.assertEqual(self.cut_plot.get_min(data, 4), 6)

    def test_change_scale_linear(self):
        self.axes.set_xscale = MagicMock()
        self.axes.set_yscale = MagicMock()
        xy_config = {'x_log': False, 'y_log': False, 'x_range': (0, 10), 'y_range': (1, 7)}

        self.cut_plot.change_axis_scale(xy_config)
        self.axes.set_xscale.assert_called_once_with('linear')
        self.axes.set_yscale.assert_called_once_with('linear')
        self.assertEqual(self.cut_plot.x_range, (0,10))
        self.assertEqual(self.cut_plot.y_range, (1,7))

    def test_change_scale_log(self):
        self.axes.set_xscale = MagicMock()
        self.axes.set_yscale = MagicMock()
        line = MagicMock()
        line.get_ydata = MagicMock(return_value=np.array([1, 5, 10]))
        line.get_xdata = MagicMock(return_value=np.array([20, 60, 12]))
        self.axes.get_lines = MagicMock(return_value = [line])
        self.canvas.figure.gca = MagicMock(return_value = self.axes)
        xy_config = {'x_log': True, 'y_log': True, 'x_range': (0, 20), 'y_range': (1, 7)}

        self.cut_plot.change_axis_scale(xy_config)
        self.axes.set_xscale.assert_called_once_with('symlog', linthreshx=10.0)
        self.axes.set_yscale.assert_called_once_with('symlog', linthreshy=1.0)
        self.assertEqual(self.cut_plot.x_range, (12,20))
        self.assertEqual(self.cut_plot.y_range, (1,7))
