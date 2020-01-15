from mock import MagicMock, patch, PropertyMock
import unittest

from matplotlib import colors
from matplotlib.legend import Legend
from mslice.plotting.plot_window.slice_plot import SlicePlot


class SlicePlotTest(unittest.TestCase):

    def setUp(self):
        self.plot_figure = MagicMock()
        self.plot_figure.window = MagicMock()
        canvas = MagicMock()
        self.plot_figure.window.canvas = canvas
        self.slice_plotter = MagicMock()
        self.axes = MagicMock()
        canvas.figure.gca = MagicMock(return_value=self.axes)
        self.slice_plot = SlicePlot(self.plot_figure, self.slice_plotter, "workspace")

    def test_that_is_changed_works_correctly(self):
        self.slice_plot.default_options = {}

        self.slice_plot.default_options['y_grid'] = False
        self.slice_plot.y_grid = False
        self.slice_plot.default_options['x_grid'] = False
        self.slice_plot.x_grid = True

        self.assertEqual(self.slice_plot.is_changed('y_grid'), False)
        self.assertEqual(self.slice_plot.is_changed('x_grid'), True)

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

    def test_lines_redrawn_on_intensity_change(self):
        self.slice_plot.save_default_options()
        self.slice_plot.toggle_overplot_line(self.slice_plot.plot_window.action_helium, 4, True, True)
        colorbar_range = PropertyMock(return_value=(0, 10))
        type(self.slice_plot).colorbar_range = colorbar_range
        self.slice_plotter.show_dynamical_susceptibility.__name__ = 'foo'
        self.slice_plot.show_intensity_plot(self.plot_figure.action_chi_qe,
                                            self.slice_plotter.show_dynamical_susceptibility,
                                            True)

        self.assertTrue(self.slice_plot.plot_window.action_helium.checked)
        self.slice_plotter.add_overplot_line.assert_any_call('workspace', 4, True, '')

    @patch('mslice.plotting.plot_window.slice_plot.QtWidgets.QInputDialog.getInt')
    def test_arbitrary_recoil_line(self, qt_get_int_mock):
        qt_get_int_mock.return_value = (5, True)
        self.slice_plotter.add_overplot_line.reset_mock()
        self.plot_figure.action_arbitrary_nuclei.isChecked = MagicMock(return_value=True)

        self.slice_plot.arbitrary_recoil_line()
        self.slice_plotter.add_overplot_line.assert_called_once_with('workspace', 5, True, None)

    @patch('mslice.plotting.plot_window.slice_plot.QtWidgets.QInputDialog.getInt')
    def test_arbitrary_recoil_line_cancelled(self, qt_get_int_mock):
        qt_get_int_mock.return_value = (5, False)
        self.slice_plotter.add_overplot_line.reset_mock()
        self.plot_figure.action_arbitrary_nuclei.isChecked = MagicMock(return_value=True)

        self.slice_plot.arbitrary_recoil_line()
        self.slice_plotter.add_overplot_line.assert_not_called()

    def test_update_legend_in_icut_mode(self):
        self.slice_plot._canvas.manager.plot_handler.icut = MagicMock()

        self.slice_plot.update_legend()

        self.axes.legend.assert_called_once()
        if hasattr(Legend, "set_draggable"):
            self.axes.legend().set_draggable.assert_called_once()
        else:
            self.axes.legend().draggable.assert_called_once()

        self.assertEqual(self.slice_plot._canvas.manager.plot_handler.icut.rect.ax, self.axes)


if __name__ == '__main__':
    unittest.main()
