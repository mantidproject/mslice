import mock
from mock import MagicMock, PropertyMock, call
import unittest
from PyQt4.QtCore import pyqtSignal
from PyQt4.QtTest import QTest
from mslice.plotting.plot_window.plot_options import CutPlotOptions, SlicePlotOptions
from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter, SlicePlotOptionsPresenter
from mslice.plotting.plot_window.plot_figure import PlotFigureManager


class PlotOptionsPresenterTest(unittest.TestCase):
    def setUp(self):
        self.view = MagicMock()
        self.model = MagicMock()

    def test_change_title(self):

        mock_view_title = PropertyMock()
        type(self.view).title = mock_view_title

        mock_model_title = PropertyMock(return_value = 'Title 0')
        type(self.model).title = mock_model_title

        #test title passed model -> view
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)

        mock_model_title.assert_called_once_with()
        mock_view_title.assert_called_once_with('Title 0')

        #test title passed view -> model
        mock_view_title.reset_mock()
        mock_model_title.reset_mock()
        mock_view_title.return_value = 'Title 1'
        self.presenter._value_modified('title')
        self.presenter.get_new_config()

        mock_view_title.assert_called_once_with()
        mock_model_title.assert_called_once_with('Title 1')

    def test_change_axis(self):

        view_x_label_mock = PropertyMock()
        view_y_label_mock = PropertyMock()
        model_x_label_mock = PropertyMock(return_value = 'x0')
        model_y_label_mock = PropertyMock(return_value = 'y0')

        type(self.view).x_label = view_x_label_mock
        type(self.view).y_label = view_y_label_mock
        type(self.model).x_label = model_x_label_mock
        type(self.model).y_label = model_y_label_mock

        #test labels passed model -> view
        self.presenter = SlicePlotOptionsPresenter(self.view, self.model)
        model_x_label_mock.assert_called_once_with()
        model_y_label_mock.assert_called_once_with()
        view_x_label_mock.assert_called_once_with('x0')
        view_y_label_mock.assert_called_once_with('y0')

        #test labels passed view -> model
        model_x_label_mock.reset_mock()
        model_y_label_mock.reset_mock()
        view_x_label_mock.reset_mock()
        view_y_label_mock.reset_mock()

        view_x_label_mock.return_value = 'x1'
        view_y_label_mock.return_value = 'y1'
        self.presenter._value_modified('x_label')
        self.presenter._value_modified('y_label')
        self.presenter.get_new_config()

        view_x_label_mock.assert_called_once_with()
        view_y_label_mock.assert_called_once_with()
        model_x_label_mock.assert_called_once_with('x1')
        model_y_label_mock.assert_called_once_with('y1')

    def test_change_xrange(self):

        view_x_range_mock = PropertyMock()
        type(self.view).x_range = view_x_range_mock

        model_x_range_mock = PropertyMock(return_value=(1, 5))
        type(self.model).x_range = model_x_range_mock

        #test passed model -> view
        self.presenter1 = SlicePlotOptionsPresenter(self.view, self.model)

        model_x_range_mock.assert_called_with()
        view_x_range_mock.assert_called_once_with((1, 5))

        #test passed view -> model through slice presenter
        view_x_range_mock.reset_mock()
        model_x_range_mock.reset_mock()
        view_x_range_mock.return_value = (2, 10)
        self.presenter1._xy_config_modified('x_range')
        self.presenter1.get_new_config()
        model_x_range_mock.assert_called_once_with((2, 10))

        #test passed view -> model through cut presenter
        self.presenter2 = CutPlotOptionsPresenter(self.view, self.model)
        model_x_range_mock.reset_mock()
        self.presenter2._xy_config_modified('x_range')
        self.presenter2.get_new_config()
        self.model.change_cut_plot.assert_called_once()








