from mock import MagicMock, PropertyMock
import unittest

import matplotlib.colors as colors
from mslice.plotting.plot_window.slice_plot import SlicePlot


class SlicePlotTest(unittest.TestCase):

    def setUp(self):
        self.plot_figure = MagicMock()
        self.canvas = MagicMock()
        self.slice_plotter = MagicMock()
        self.axes = MagicMock()
        self.canvas.figure.gca = MagicMock(return_value=self.axes)
        self.slice_plot = SlicePlot(self.plot_figure, self.canvas, self.slice_plotter)

    def test_change_logarithmic(self):
        image = MagicMock()
        norm = PropertyMock(return_value = colors.Normalize)
        type(image).norm = norm
        self.axes.get_images = MagicMock(return_value=[image])
        self.slice_plot.change_slice_plot((0,10), True)
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
        self.axes.get_images = MagicMock(return_value=[image])
        self.slice_plot.change_slice_plot((0, 15), False)
        image.set_norm.assert_called_once()
        norm_set = image.set_norm.call_args_list[0][0][0]
        self.assertEqual(norm_set.vmin, 0)
        self.assertEqual(norm_set.vmax, 15)
        self.assertEqual(type(norm_set), colors.Normalize)
        image.set_clim.assert_called_once_with((0, 15))
