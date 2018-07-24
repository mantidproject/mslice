from mock import MagicMock, patch, PropertyMock, ANY
import unittest

from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.text import Text
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

    def test_change_logarithmic(self):
        image = MagicMock()
        norm = PropertyMock(return_value=colors.Normalize)
        type(image).norm = norm
        self.axes.get_images = MagicMock(return_value=[image])
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
        self.axes.get_images = MagicMock(return_value=[image])
        self.slice_plot.change_axis_scale((0, 15), False)
        image.set_norm.assert_called_once()
        norm_set = image.set_norm.call_args_list[0][0][0]
        self.assertEqual(norm_set.vmin, 0)
        self.assertEqual(norm_set.vmax, 15)
        self.assertEqual(type(norm_set), colors.Normalize)
        image.set_clim.assert_called_once_with((0, 15))

    def test_reset_checkboxes(self):
        line1 = MagicMock()
        line2 = MagicMock()
        line1.get_linestyle = MagicMock(return_value='None')
        line2.get_linestyle = MagicMock(return_value='-')
        self.slice_plotter.overplot_lines.__getitem__ = MagicMock(return_value={0: line1, 1: line2})
        self.slice_plotter.get_recoil_label = MagicMock()
        self.slice_plotter.get_recoil_label.side_effect = ['0', '1']
        self.slice_plot.reset_info_checkboxes()
        self.slice_plot.plot_window.disable_action.assert_called_once_with('0')

    def test_lines_redrawn(self):
        self.slice_plot.toggle_overplot_line(self.slice_plot.plot_window.action_helium, 4, True, True)
        new_slice_plot = SlicePlot(self.plot_figure, self.slice_plotter, "workspace")

        self.assertTrue(new_slice_plot.plot_window.action_helium.checked)
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

    def test_update_legend(self):
        line1 = Line2D([], [],label='line1')
        line2 = Line2D([], [], label='line2')
        line3 = Line2D([], [], label='line_not_to_show1')
        line4 = Line2D([], [], label='')
        line3.set_linestyle('None')
        line1_text = Text('line1')
        line2_text = Text('line2')

        get_legend_mock = MagicMock()
        get_legend_mock.get_lines = MagicMock(return_value=[line1, line2])
        get_legend_mock.get_texts = MagicMock(return_value=[line1_text, line2_text])
        self.axes.get_children = MagicMock(return_value=[line1, line2, line3, line4, "nonsense", 4])
        self.axes.legend = MagicMock(return_value=get_legend_mock)

        self.slice_plot.update_legend()
        self.axes.legend.assert_called_with([line1, line2], ['line1', 'line2'], fontsize=ANY)
        self.assertEqual(self.slice_plot._legend_dict[line1], line1)
        self.assertEqual(self.slice_plot._legend_dict[line2], line2)
        self.assertEqual(self.slice_plot._legend_dict[line1_text], line1)
        self.assertEqual(self.slice_plot._legend_dict[line2_text], line2)
