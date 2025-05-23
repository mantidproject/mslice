from mock import MagicMock, PropertyMock, Mock, ANY
import unittest
from mslice.presenters.plot_options_presenter import (
    CutPlotOptionsPresenter,
    SlicePlotOptionsPresenter,
)


class PlotOptionsPresenterTest(unittest.TestCase):
    def setUp(self):
        self.view = MagicMock()
        self.model = MagicMock()

    def test_change_title(self):
        mock_view_title = PropertyMock()
        type(self.view).title = mock_view_title

        mock_model_title = PropertyMock(return_value="Title 0")
        type(self.model).title = mock_model_title

        # title passed model -> view
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)

        mock_model_title.assert_called_once_with()
        mock_view_title.assert_called_once_with("Title 0")

        # title passed view -> model
        mock_view_title.reset_mock()
        mock_model_title.reset_mock()
        mock_view_title.return_value = "Title 1"
        self.presenter._value_modified("title")
        self.presenter.get_new_config()

        mock_view_title.assert_called_once_with()
        mock_model_title.assert_called_once_with("Title 1")

    def test_change_axis(self):
        view_x_label_mock = PropertyMock()
        view_y_label_mock = PropertyMock()
        model_x_label_mock = PropertyMock(return_value="x0")
        model_y_label_mock = PropertyMock(return_value="y0")

        type(self.view).x_label = view_x_label_mock
        type(self.view).y_label = view_y_label_mock
        type(self.model).x_label = model_x_label_mock
        type(self.model).y_label = model_y_label_mock

        # labels passed model -> view
        self.presenter = SlicePlotOptionsPresenter(self.view, self.model)
        model_x_label_mock.assert_called_once_with()
        model_y_label_mock.assert_called_once_with()
        view_x_label_mock.assert_called_once_with("x0")
        view_y_label_mock.assert_called_once_with("y0")

        # labels passed view -> model
        model_x_label_mock.reset_mock()
        model_y_label_mock.reset_mock()
        view_x_label_mock.reset_mock()
        view_y_label_mock.reset_mock()

        view_x_label_mock.return_value = "x1"
        view_y_label_mock.return_value = "y1"
        self.presenter._value_modified("x_label")
        self.presenter._value_modified("y_label")
        self.presenter.get_new_config()

        view_x_label_mock.assert_called_once_with()
        view_y_label_mock.assert_called_once_with()
        model_x_label_mock.assert_called_once_with("x1")
        model_y_label_mock.assert_called_once_with("y1")

    def test_change_grid(self):
        view_x_grid_mock = PropertyMock()
        view_y_grid_mock = PropertyMock()
        model_x_grid_mock = PropertyMock(return_value=False)
        model_y_grid_mock = PropertyMock(return_value=False)

        type(self.view).x_grid = view_x_grid_mock
        type(self.view).y_grid = view_y_grid_mock
        type(self.model).x_grid = model_x_grid_mock
        type(self.model).y_grid = model_y_grid_mock

        # labels passed model -> view
        self.presenter = SlicePlotOptionsPresenter(self.view, self.model)
        model_x_grid_mock.assert_called_once_with()
        model_y_grid_mock.assert_called_once_with()
        view_x_grid_mock.assert_called_once_with(False)
        view_y_grid_mock.assert_called_once_with(False)

        # labels passed view -> model
        model_x_grid_mock.reset_mock()
        model_y_grid_mock.reset_mock()
        view_x_grid_mock.reset_mock()
        view_y_grid_mock.reset_mock()

        view_x_grid_mock.return_value = True
        view_y_grid_mock.return_value = True
        self.presenter._value_modified("x_grid")
        self.presenter._value_modified("y_grid")
        self.presenter.get_new_config()

        view_x_grid_mock.assert_called_once_with()
        view_y_grid_mock.assert_called_once_with()
        model_x_grid_mock.assert_called_once_with(True)
        model_y_grid_mock.assert_called_once_with(True)

    def test_change_xrange(self):
        view_x_range_mock = PropertyMock()
        type(self.view).x_range = view_x_range_mock

        model_x_range_mock = PropertyMock(return_value=(1, 5))
        type(self.model).x_range = model_x_range_mock

        # passed model -> view
        self.presenter1 = SlicePlotOptionsPresenter(self.view, self.model)

        model_x_range_mock.assert_called_with()
        view_x_range_mock.assert_called_once_with((1, 5))

        # passed view -> model through slice presenter
        model_x_range_mock.reset_mock()
        view_x_range_mock.return_value = (2, 10)
        self.presenter1._value_modified("x_range")
        self.presenter1.get_new_config()
        model_x_range_mock.assert_called_once_with((2, 10))

        # passed view -> model through cut presenter
        model_x_range_mock.reset_mock()
        view_x_range_mock.return_value = (3, 11)
        self.presenter2 = CutPlotOptionsPresenter(self.view, self.model)
        self.presenter2._value_modified("x_range")
        self.presenter2.get_new_config()
        model_x_range_mock.assert_called_with((3, 11))

    def test_change_colorbar_config(self):
        view_colorbar_range_mock = PropertyMock()
        view_colorbar_log_mock = PropertyMock()
        type(self.view).colorbar_range = view_colorbar_range_mock
        type(self.view).colorbar_log = view_colorbar_log_mock

        model_colorbar_range_mock = PropertyMock(return_value=(1, 5))
        model_colorbar_log_mock = PropertyMock(return_value=False)
        type(self.model).colorbar_range = model_colorbar_range_mock
        type(self.model).colorbar_log = model_colorbar_log_mock

        # passed model -> view
        self.presenter = SlicePlotOptionsPresenter(self.view, self.model)

        model_colorbar_range_mock.assert_called_with()
        model_colorbar_log_mock.assert_called_with()
        view_colorbar_range_mock.assert_called_once_with((1, 5))
        view_colorbar_log_mock.assert_called_once_with(False)

        # passed view -> model
        view_colorbar_range_mock.return_value = (2, 10)
        view_colorbar_log_mock.return_value = True
        self.presenter._set_c_range()
        self.presenter._set_colorbar_log()
        self.presenter.get_new_config()
        self.model.change_axis_scale.assert_called_once_with((2, 10), True)

    def test_change_xy_log(self):
        view_x_log_mock = PropertyMock()
        view_y_log_mock = PropertyMock()
        model_x_log_mock = PropertyMock(return_value=True)
        model_y_log_mock = PropertyMock(return_value=False)
        type(self.view).x_log = view_x_log_mock
        type(self.view).y_log = view_y_log_mock
        type(self.model).x_log = model_x_log_mock
        type(self.model).y_log = model_y_log_mock

        # model -> view
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)
        view_x_log_mock.assert_called_once_with(True)
        view_y_log_mock.assert_called_once_with(False)

        # view -> model
        view_x_log_mock.return_value = False
        view_y_log_mock.return_value = True
        self.presenter._value_modified("x_log")
        self.presenter._value_modified("y_log")
        self.presenter.get_new_config()
        model_x_log_mock.assert_called_with(False)
        model_y_log_mock.assert_called_with(True)

    def test_line_options(self):
        #  model -> view
        line_options = [
            {
                "color": "k",
                "style": "-",
                "width": "10",
                "marker": "*",
                "error_bars": True,
            }
        ]
        legends = {"label": "legend1", "visible": True}
        line_data = list(zip(legends, line_options))
        self.model.get_all_line_options = Mock(return_value=line_data)
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)

        self.model.get_all_line_options.assert_called_once_with()
        self.view.set_line_options.assert_called_once_with(line_data)

        #  view -> model
        line_options2 = [
            {
                "color": "b",
                "style": "-",
                "width": "10",
                "marker": "o",
                "error_bars": False,
            }
        ]
        legends2 = {"label": "legend1", "visible": False}
        line_data2 = list(zip(legends2, line_options2))

        self.view.get_line_options = Mock(return_value=line_data2)
        self.presenter.get_new_config()

        self.view.get_line_options.assert_called_once_with()
        self.model.set_all_line_options.assert_called_once_with(line_data2, ANY)

    def test_remove_line(self):
        self.remove_line_by_index = Mock()
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)

        # check line with correct index removed
        self.presenter.remove_container(9)
        self.model.remove_line_by_index.assert_called_once_with(9)

    def test_set_all_fonts_size(self):
        model_all_fonts_size = PropertyMock()
        view_all_fonts_size = PropertyMock()
        type(self.model).all_fonts_size = model_all_fonts_size
        type(self.view).all_fonts_size = view_all_fonts_size

        self.presenter = CutPlotOptionsPresenter(self.view, self.model)
        fonts_config = {
            "title_size": 15,
            "x_range_font_size": 14,
            "y_range_font_size": 13,
            "x_label_size": 12,
            "y_label_size": 11,
        }
        self.presenter._default_font_sizes_config = fonts_config

        # view -> model
        view_all_fonts_size.return_value = 20
        self.presenter._set_all_plot_fonts()

        view_all_fonts_size.assert_called_once_with()

        fonts_updated = {
            "title_size": 20,
            "x_range_font_size": 20,
            "y_range_font_size": 20,
            "x_label_size": 20,
            "y_label_size": 20,
        }
        model_all_fonts_size.assert_any_call(
            fonts_updated
        )  # Not latest call due to copy() methods
