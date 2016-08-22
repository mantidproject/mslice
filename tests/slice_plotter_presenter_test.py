import unittest

import mock

from mainview import MainView
from models.slice.slice_plotter import SlicePlotter
from presenters.interfaces.main_presenter import MainPresenterInterface
from presenters.slice_plotter_presenter import SlicePlotterPresenter,Axis
from views.slice_plotter_view import SlicePlotterView
from widgets.slice.command import Command


class SlicePlotterPresenterTest(unittest.TestCase):
    def setUp(self):
        # Setting up main tool presenter and view
        self.main_view = mock.create_autospec(MainView)
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        self.main_view.get_presenter = mock.Mock(return_value=self.main_presenter)
        # Setting up view for slice plotter window
        self.slice_plotter = mock.create_autospec(SlicePlotter)
        self.slice_view = mock.create_autospec(SlicePlotterView)

    def test_constructor_success(self):
        self.slice_plotter_presenter = SlicePlotterPresenter(self.slice_view, self.slice_plotter)


    def test_constructor_invalid_main_view_failure(self):
        self.assertRaises(TypeError, SlicePlotterPresenter, self.slice_view, self.slice_view, self.slice_plotter)

    def test_constructor_invalid_slice_view_failure(self):
        self.assertRaises(TypeError, SlicePlotterPresenter, self.main_view, self.main_view, self.slice_plotter)

    def test_constructor_invalid_slice_plotter_failure(self):
        self.assertRaises(TypeError, SlicePlotterPresenter, self.main_view, self.slice_view, self.slice_view)

    def test_notify_presenter_unknown_command_raise_exception_failure(self):
        self.slice_plotter_presenter = SlicePlotterPresenter(self.slice_view, self.slice_plotter)
        unknown_command = -1
        self.assertRaises(ValueError, self.slice_plotter_presenter.notify, unknown_command)

    def test_register_master_success(self):
        slice_presenter = SlicePlotterPresenter(self.slice_view, self.slice_plotter)
        slice_presenter.register_master(self.main_view)
        self.main_presenter.subscribe_to_workspace_selection_monitor.assert_called_once_with(slice_presenter)

    def test_register_master_invalid_master_fail(self):
        slice_presenter = SlicePlotterPresenter(self.slice_view, self.slice_plotter)
        self.assertRaises(AssertionError, slice_presenter.register_master, 3)

    def test_plot_slice_successful(self):
        self.slice_plotter_presenter = SlicePlotterPresenter(self.slice_view, self.slice_plotter)
        self.slice_plotter_presenter.register_master(self.main_view)
        x = Axis('x', 0, 10 ,1)
        y = Axis('y', 2, 8, 3)
        intensity_start = 7
        intensity_end = 8
        norm_to_one = 9
        smoothing = 10
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value=[selected_workspace]
        self.slice_view.get_slice_x_axis.return_value=x.units
        self.slice_view.get_slice_x_start.return_value=x.start
        self.slice_view.get_slice_x_end.return_value=x.end
        self.slice_view.get_slice_x_step.return_value=x.step
        self.slice_view.get_slice_y_axis.return_value=y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value=y.end
        self.slice_view.get_slice_y_step.return_value=y.step
        self.slice_view.get_slice_intensity_start.return_value=intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value=norm_to_one
        self.slice_view.get_slice_smoothing.return_value=smoothing
        self.slice_view.get_slice_colourmap.return_value=colourmap

        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        self.main_presenter.get_selected_workspaces.assert_called_once_with()
        self.slice_view.get_slice_x_axis.assert_called_once_with()
        self.slice_view.get_slice_x_start.assert_called_once_with()
        self.slice_view.get_slice_x_end.assert_called_once_with()
        self.slice_view.get_slice_x_step.assert_called_once_with()
        self.slice_view.get_slice_y_axis.assert_called_once_with()
        self.slice_view.get_slice_y_start.assert_called_once_with()
        self.slice_view.get_slice_y_end.assert_called_once_with()
        self.slice_view.get_slice_y_step.assert_called_once_with()
        self.slice_view.get_slice_intensity_start.assert_called_once_with()
        self.slice_view.get_slice_intensity_end.assert_called_once_with()
        self.slice_view.get_slice_is_norm_to_one.assert_called_once_with()
        self.slice_view.get_slice_smoothing.assert_called_once_with()
        self.slice_view.get_slice_colourmap.assert_called_once_with()

        self.slice_plotter.display_slice.assert_called_with(selected_workspace=selected_workspace, x_axis=x, y_axis=y, smoothing=smoothing,
                                                            intensity_start=intensity_start,intensity_end=intensity_end,
                                                            norm_to_one=norm_to_one,  colourmap=colourmap)

