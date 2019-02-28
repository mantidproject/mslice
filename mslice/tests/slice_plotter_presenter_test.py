from __future__ import (absolute_import, division, print_function)
import mock
import unittest

from mslice.models.axis import Axis
from mslice.models.cmap import DEFAULT_CMAP
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
    @mock.patch('mslice.presenters.slice_plotter_presenter.create_slice_figure')
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
        slice_presenter.plot_slice('workspace', x_axis, y_axis, 7, 8, False, DEFAULT_CMAP)
        self.assertTrue('workspace' in slice_presenter._slice_cache)
        create_slice_mock.assert_called_once()
        plot_cached_slice_mock.assert_called_once()

    @mock.patch('mslice.presenters.slice_plotter_presenter.plot_overplot_line')
    @mock.patch('mslice.presenters.slice_plotter_presenter.compute_recoil_line')
    @mock.patch('mslice.presenters.slice_plotter_presenter.compute_powder_line')
    def test_add_overplot_line(self, compute_powder_mock, compute_recoil_mock, plot_line_mock):
        slice_presenter = SlicePlotterPresenter()
        compute_recoil_mock.return_value = ('compute', 'recoil')
        compute_powder_mock.return_value = ('compute', 'powder')
        plot_line_mock.return_value = 'plot'
        recoil_key = 5
        powder_key = 6
        cache_mock = mock.MagicMock()
        slice_presenter._slice_cache['workspace'] = cache_mock
        type(cache_mock).overplot_lines = mock.MagicMock()
        cache_mock.energy_axis.e_unit = 'meV'

        slice_presenter.add_overplot_line('workspace', recoil_key, True)
        compute_recoil_mock.assert_called_once()
        cache_mock.overplot_lines.__setitem__.assert_called_with(recoil_key, 'plot')
        plot_line_mock.assert_called_with('compute', 'recoil', recoil_key, True, cache_mock)
        slice_presenter.add_overplot_line('workspace', powder_key, False)
        compute_powder_mock.assert_called_once()
        cache_mock.overplot_lines.__setitem__.assert_called_with(powder_key, 'plot')
        plot_line_mock.assert_called_with('compute', 'powder', powder_key, False, cache_mock)

    @mock.patch('mslice.presenters.slice_plotter_presenter.remove_line')
    def test_hide_overplot_line(self, remove_line_mock):
        slice_presenter = SlicePlotterPresenter()
        cache_mock = mock.MagicMock()
        key = 5
        type(cache_mock).overplot_lines = {key: 'line'}
        slice_presenter._slice_cache['workspace'] = cache_mock
        slice_presenter.hide_overplot_line('workspace', key)
        self.assertTrue(key not in cache_mock.overplot_lines)
        remove_line_mock.assert_called_once_with('line')
