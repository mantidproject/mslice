from __future__ import (absolute_import, division, print_function)
import mock
import unittest

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
