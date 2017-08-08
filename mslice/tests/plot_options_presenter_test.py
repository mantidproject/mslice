from mock import MagicMock, PropertyMock
import unittest
from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter, SlicePlotOptionsPresenter


class PlotOptionsPresenterTest(unittest.TestCase):
    def setUp(self):
        self.view = MagicMock()
        self.model = MagicMock()

    def test_change_title(self):

        mock_view_title = PropertyMock()
        type(self.view).title = mock_view_title

        mock_model_title = PropertyMock(return_value='Title 0')
        type(self.model).title = mock_model_title

        # title passed model -> view
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)

        mock_model_title.assert_called_once_with()
        mock_view_title.assert_called_once_with('Title 0')

        # title passed view -> model
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
        model_x_label_mock = PropertyMock(return_value='x0')
        model_y_label_mock = PropertyMock(return_value='y0')

        type(self.view).x_label = view_x_label_mock
        type(self.view).y_label = view_y_label_mock
        type(self.model).x_label = model_x_label_mock
        type(self.model).y_label = model_y_label_mock

        # labels passed model -> view
        self.presenter = SlicePlotOptionsPresenter(self.view, self.model)
        model_x_label_mock.assert_called_once_with()
        model_y_label_mock.assert_called_once_with()
        view_x_label_mock.assert_called_once_with('x0')
        view_y_label_mock.assert_called_once_with('y0')

        # labels passed view -> model
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

        # passed model -> view
        self.presenter1 = SlicePlotOptionsPresenter(self.view, self.model)

        model_x_range_mock.assert_called_with()
        view_x_range_mock.assert_called_once_with((1, 5))

        # passed view -> model through slice presenter
        model_x_range_mock.reset_mock()
        view_x_range_mock.return_value = (2, 10)
        self.presenter1._xy_config_modified('x_range')
        self.presenter1.get_new_config()
        model_x_range_mock.assert_called_once_with((2, 10))

        # passed view -> model through cut presenter
        self.presenter2 = CutPlotOptionsPresenter(self.view, self.model)
        self.presenter2._xy_config_modified('x_range')
        self.presenter2.get_new_config()
        self.model.change_cut_plot.assert_called_once()

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
        self.model.change_slice_plot.assert_called_once_with((2, 10), True)

    def test_change_xy_log(self):
        view_x_log_mock = PropertyMock()
        view_y_log_mock = PropertyMock()
        model_x_log_mock = PropertyMock(return_value=True)
        model_y_log_mock = PropertyMock(return_value=False)
        model_x_range_mock = PropertyMock(return_value=(1, 2))
        model_y_range_mock = PropertyMock(return_value=(3, 4))
        type(self.view).x_log = view_x_log_mock
        type(self.view).y_log = view_y_log_mock
        type(self.model).x_log = model_x_log_mock
        type(self.model).y_log = model_y_log_mock
        type(self.model).x_range = model_x_range_mock
        type(self.model).y_range = model_y_range_mock

        # model -> view
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)
        view_x_log_mock.assert_called_once_with(True)
        view_y_log_mock.assert_called_once_with(False)

        # view -> model
        view_x_log_mock.return_value = False
        view_y_log_mock.return_value = True
        self.presenter._xy_config_modified('x_log')
        self.presenter._xy_config_modified('y_log')
        self.presenter.get_new_config()
        self.model.change_cut_plot.assert_called_once_with({'x_range': (1, 2), 'y_range': (3, 4), 'modified': True,
                                                            'x_log': False,    'y_log': True})

    def test_show_error_bars(self):
        view_error_bars_mock = PropertyMock()
        model_error_bars_mock = PropertyMock(return_value=True)
        type(self.view).error_bars = view_error_bars_mock
        type(self.model).error_bars = model_error_bars_mock

        # model -> view
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)
        model_error_bars_mock.assert_called_once_with()
        view_error_bars_mock.assert_called_once_with(True)

        # view -> model
        model_error_bars_mock.reset_mock()
        view_error_bars_mock.return_value = False
        self.presenter._value_modified('error_bars')
        self.presenter.get_new_config()

        model_error_bars_mock.assert_called_once_with(False)

    '''TODO: add/update tests
    def test_legends(self):
        #  model -> view
        legends = LegendDescriptor(handles=['legend0', 'legend1'])
        self.model.get_legends = Mock(return_value=legends)
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)

        self.model.get_legends.assert_called_once_with()
        self.view.set_legends.assert_called_once_with(legends)

        #  view -> model
        legends2 = LegendDescriptor(visible=False, handles=['legend2'])
        self.view.get_legends = Mock(return_value=legends2)
        self.presenter.get_new_config()

        self.view.get_legends.assert_called_once_with()
        self.model.set_legends.assert_called_once_with(legends2)'''
