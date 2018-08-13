from __future__ import (absolute_import, division, print_function)
import mock
import unittest

from mslice.models.axis import Axis
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter


class SlicePlotterPresenterTest(unittest.TestCase):

    def test_register_master_success(self):
        slice_presenter = SlicePlotterPresenter()
        main_presenter = mock.create_autospec(MainPresenterInterface)
        slice_presenter.register_master(main_presenter)
        main_presenter.subscribe_to_workspace_selection_monitor.assert_called_once_with(slice_presenter)

    def test_validate_intensity_success(self):
        slice_presenter = SlicePlotterPresenter()
        start, end = slice_presenter.validate_intensity('7', '8')
        self.assertEqual(start, 7.0)
        self.assertEqual(end, 8.0)

    def test_validate_intensity_erroneous_type_fail(self):
        slice_presenter = SlicePlotterPresenter()
        with self.assertRaises(ValueError):
            slice_presenter.validate_intensity('j', '8')


    def test_validate_intensity_end_less_than_intensity_start_fail(self):
        slice_presenter = SlicePlotterPresenter()
        with self.assertRaises(ValueError):
            slice_presenter.validate_intensity('8', '7')

    @mock.patch('mslice.presenters.slice_plotter_presenter.plot_cached_slice')
    @mock.patch('mslice.presenters.slice_plotter_presenter.create_slice')
    @mock.patch('mslice.presenters.slice_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.presenters.slice_plotter_presenter.sample_temperature')
    @mock.patch('mslice.presenters.slice_plotter_presenter.compute_slice')
    def test_plot_slice_success(self, compute_slice_mock, sample_temp_mock, get_workspace_handle_mock,
                                create_slice_mock, plot_cached_slice_mock):
        workspace_mock = mock.MagicMock()
        name = mock.PropertyMock(return_value='workspace')
        type(workspace_mock).name = name
        slice_mock = mock.MagicMock()
        slice_name = mock.PropertyMock(return_value='__workspace')
        type(slice_mock).name = slice_name
        get_workspace_handle_mock.return_value = workspace_mock
        sample_temp_mock.return_value = 5
        compute_slice_mock.return_value = slice_mock
        x_axis = Axis('x', 0, 1, 0.1)
        y_axis = Axis('y', 0, 1, 0.1)
        slice_presenter = SlicePlotterPresenter()
        slice_presenter.plot_slice('workspace', x_axis, y_axis, 7, 8, False, 'viridis')
        self.assertTrue('workspace' in slice_presenter._slice_cache)
        create_slice_mock.assert_called_once()
        plot_cached_slice_mock.assert_called_once()
