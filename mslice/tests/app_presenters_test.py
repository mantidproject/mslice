import unittest
import mock
from mslice.app.presenters import (is_gui, get_cut_plotter_presenter, get_slice_plotter_presenter,
                                   get_dataloader_presenter, get_powder_presenter, cli_dataloader_presenter,
                                   cli_slice_plotter_presenter, cli_powder_presenter, cli_cut_plotter_presenter)


class AppPresentersTest(unittest.TestCase):

    @mock.patch('mslice.app.presenters.MAIN_WINDOW')
    def test_that_is_gui_works_as_expected_with_gui(self, MAIN_WINDOW):
        MAIN_WINDOW = not None  # noqa: 841
        self.assertTrue(is_gui())

    def test_that_is_gui_works_as_expected_without_gui(self):
        self.assertFalse(is_gui())

    @mock.patch('mslice.app.presenters.MAIN_WINDOW')
    @mock.patch('mslice.app.presenters.is_gui')
    def test_that_get_data_loader_presenter_works_as_expected(self, is_gui, MAIN_WINDOW):
        is_gui.return_value = False
        return_value = get_dataloader_presenter()
        self.assertEquals(return_value, cli_dataloader_presenter)

        is_gui.return_value = True
        MAIN_WINDOW.dataloader_presenter = cli_dataloader_presenter
        return_value = get_dataloader_presenter()
        self.assertEquals(return_value, cli_dataloader_presenter)

    @mock.patch('mslice.app.presenters.MAIN_WINDOW')
    @mock.patch('mslice.app.presenters.is_gui')
    def test_that_get_slice_plotter_presenter_works_as_expected(self, is_gui, MAIN_WINDOW):
        is_gui.return_value = False
        return_value = get_slice_plotter_presenter()
        self.assertEquals(return_value, cli_slice_plotter_presenter)

        is_gui.return_value = True
        MAIN_WINDOW.slice_plotter_presenter = cli_slice_plotter_presenter
        return_value = get_slice_plotter_presenter()
        self.assertEquals(return_value, cli_slice_plotter_presenter)

    @mock.patch('mslice.app.presenters.MAIN_WINDOW')
    @mock.patch('mslice.app.presenters.is_gui')
    def test_that_get_cut_plotter_presenter_works_as_expected(self, is_gui, MAIN_WINDOW):
        is_gui.return_value = False
        return_value = get_cut_plotter_presenter()
        self.assertEquals(return_value, cli_cut_plotter_presenter)

        is_gui.return_value = True
        MAIN_WINDOW.cut_plotter_presenter = cli_cut_plotter_presenter
        return_value = get_cut_plotter_presenter()
        self.assertEquals(return_value, cli_cut_plotter_presenter)

    @mock.patch('mslice.app.presenters.MAIN_WINDOW')
    @mock.patch('mslice.app.presenters.is_gui')
    def test_that_get_powder_presenter_works_as_expected(self, is_gui, MAIN_WINDOW):
        is_gui.return_value = False
        return_value = get_powder_presenter()
        self.assertEquals(return_value, cli_powder_presenter)

        is_gui.return_value = True
        MAIN_WINDOW.powder_presenter = cli_powder_presenter
        return_value = get_powder_presenter()
        self.assertEquals(return_value, cli_powder_presenter)
