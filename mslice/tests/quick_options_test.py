from mock import MagicMock, PropertyMock, Mock, patch
import unittest

from mslice.presenters.quick_options_presenter import quick_options, QuickLinePresenter, QuickAxisPresenter, QuickLabelPresenter
from mslice.plotting.plot_window.quick_options import QuickAxisOptions, QuickLabelOptions, QuickLineOptions

class QuickOptionsTest(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()

    @patch.object(QuickLabelOptions, '__init__', lambda x, y: None)
    @patch.object(QuickLabelOptions, 'exec_', lambda x: None)
    def test_label(self):
        from matplotlib import text
        self.target = Mock(spec=text.Text)
        self.presenter = quick_options(self.target, self.model)
        assert type(self.presenter) is QuickLabelPresenter

    @patch.object(QuickLineOptions, '__init__', lambda x, y: None)
    @patch.object(QuickLineOptions, 'exec_', lambda x: None)
    def test_line(self):
        self.target = 1
        self.presenter = quick_options(self.target, self.model)
        assert type(self.presenter) is QuickLinePresenter

    @patch.object(QuickAxisOptions, '__init__', lambda v, w, x, y, z: None)
    @patch.object(QuickAxisOptions, 'exec_', lambda x: True)
    @patch.object(QuickAxisOptions, 'range_min', PropertyMock(return_value='0'))
    @patch.object(QuickAxisOptions, 'range_max', PropertyMock(return_value='10'))
    @patch.object(QuickAxisOptions, 'grid_state', PropertyMock(return_value=True))
    def test_axis_with_grid(self):
        self.target = 'y_range'
        self.presenter = quick_options(self.target, self.model)
        assert type(self.presenter) is QuickAxisPresenter
        self.assertEquals(self.model.y_grid, True)

    @patch.object(QuickAxisOptions, '__init__', lambda v, w, x, y, z: None)
    @patch.object(QuickAxisOptions, 'exec_', lambda x: True)
    @patch.object(QuickAxisOptions, 'range_min', PropertyMock(return_value='0'))
    @patch.object(QuickAxisOptions, 'range_max', PropertyMock(return_value='10'))
    def test_axis_no_grid(self):
        self.target = 'colorbar_range'
        self.presenter = quick_options(self.target, self.model)
        assert type(self.presenter) is QuickAxisPresenter


class QuickAxisTest(unittest.TestCase):

    def setUp(self):
        self.view = MagicMock()
        self.model = MagicMock()
        self.model.canvas.draw = MagicMock()
        range_min = PropertyMock(return_value=5)
        type(self.view).range_min = range_min
        range_max = PropertyMock(return_value=10)
        type(self.view).range_max = range_max
        grid_state = PropertyMock(return_value=True)
        type(self.view).grid_state = grid_state

    def test_accept(self):
        self.view.exec_ = MagicMock(return_value=True)
        QuickAxisPresenter(self.view, 'x_range', self.model, False, None)
        self.assertEquals(self.model.x_range, (5,10))
        self.assertEquals(self.model.x_grid, True)

    def test_reject(self):
        self.view.exec_ = MagicMock(return_value=False)
        self.view.set_range = Mock()
        QuickAxisPresenter(self.view, 'x_range', self.model, False, None)
        self.view.set_range.assert_not_called()
        self.view.set_grid.assert_not_called()

    def test_colorbar(self):
        self.view.exec_ = MagicMock(return_value=True)
        colorbar_log = PropertyMock()
        type(self.model).colorbar_log = colorbar_log
        self.view.log_scale.isChecked = Mock()
        QuickAxisPresenter(self.view, 'colorbar_range', self.model, None, True)
        self.view.log_scale.isChecked.assert_called_once()
        colorbar_log.assert_called_once()


class QuickLabelTest(unittest.TestCase):

    def setUp(self):
        self.view = MagicMock()
        self.model = MagicMock()
        self.target = MagicMock()
        label = PropertyMock(return_value="label")
        type(self.view).label = label
        self.target.set_text = MagicMock()

    def test_accept(self):
        self.view.exec_ = MagicMock(return_value=True)
        QuickLabelPresenter(self.view, self.target, self.model)
        self.target.set_text.assert_called_once_with("label")

    def test_reject(self):
        self.view.exec_ = MagicMock(return_value=False)
        QuickLabelPresenter(self.view, self.target, self.model)
        self.target.set_text.assert_not_called()


class QuickLineTest(unittest.TestCase):

    def setUp(self):
        self.view = MagicMock()
        self.model = MagicMock()
        self.target = MagicMock()
        self.model.canvas.draw = MagicMock()
        self.target.set_color = MagicMock()
        self.target.set_linestyle = MagicMock()
        self.target.set_linewidth = MagicMock()
        self.target.set_marker = MagicMock()
        self.target.set_label = MagicMock()
        color = PropertyMock(return_value=1)
        type(self.view).color = color
        style = PropertyMock(return_value=2)
        type(self.view).style = style
        width = PropertyMock(return_value=3)
        type(self.view).width = width
        marker = PropertyMock(return_value=4)
        type(self.view).marker = marker
        label = PropertyMock(return_value=5)
        type(self.view).label = label

    def test_accept(self):
        shown = PropertyMock(return_value=True)
        type(self.view).shown = shown
        legend = PropertyMock(return_value=True)
        type(self.view).legend = legend
        self.view.exec_ = MagicMock(return_value=True)
        QuickLinePresenter(self.view, self.target, self.model)

    def test_accept_legend_shown(self):
        shown = PropertyMock(return_value=False)
        type(self.view).shown = shown
        legend = PropertyMock(return_value=False)
        type(self.view).legend = legend
        self.view.exec_ = MagicMock(return_value=True)
        QuickLinePresenter(self.view, self.target, self.model)
        values = {'color': 1, 'style': 2,'width': 3, 'marker':4, 'label':5, 'shown':False, 'legend':False}
        self.model.set_line_data.assert_called_once_with(self.target, values)

    def test_reject(self):
        self.view.exec_ = MagicMock(return_value=False)
        QuickLinePresenter(self.view, self.target, self.model)
        self.model.set_line_data.assert_not_called()
