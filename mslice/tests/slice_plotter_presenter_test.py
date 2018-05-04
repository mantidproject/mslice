from __future__ import (absolute_import, division, print_function)
import mock
import unittest

from mslice.models.axis import Axis
from mslice.models.slice.slice_plotter import SlicePlotter
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.views.slice_plotter_view import SlicePlotterView
from mslice.widgets.slice.command import Command


class SlicePlotterPresenterTest(unittest.TestCase):
    def setUp(self):
        # Setting up main tool presenter and view
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
        # Setting up view for slice plotter window
        self.slice_plotter = mock.create_autospec(SlicePlotter)
        self.slice_view = mock.create_autospec(SlicePlotterView)

    def test_constructor_success(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )

    def test_constructor_invalid_slice_view_failure(self):
        self.assertRaises(TypeError, SlicePlotterPresenter, self.slice_plotter, self.slice_plotter)

    def test_constructor_invalid_slice_plotter_failure(self):
        self.assertRaises(TypeError, SlicePlotterPresenter, self.slice_view, self.slice_view)

    def test_notify_presenter_unknown_command_raise_exception_failure(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        unknown_command = -1
        self.assertRaises(ValueError, self.slice_plotter_presenter.notify, unknown_command)

    def test_register_master_success(self):
        slice_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        slice_presenter.register_master(self.main_presenter)
        self.main_presenter.subscribe_to_workspace_selection_monitor.assert_called_once_with(slice_presenter)

    def test_register_master_invalid_master_fail(self):
        slice_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.assertRaises(AssertionError, slice_presenter.register_master, 3)

    def test_plot_slice_successful(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        self.slice_plotter_presenter.register_master(self.main_presenter)
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x', '0', '10' ,'1')
        y = Axis('y', '2', '8', '3')
        intensity_start = '7'
        intensity_end = '8'
        norm_to_one = False
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap
        plot_info = ("plot_data", "boundaries", "colormap", "norm")
        self.slice_plotter.plot_slice = mock.Mock( return_value=plot_info )

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

        self.slice_plotter.plot_slice.assert_called_with( selected_workspace, Axis( 'x', 0, 10, 1 ),
                                                          Axis('y', 2, 8, 3), int(smoothing),
                                                          float(intensity_start), float(intensity_end),
                                                          norm_to_one, colourmap)

    def test_plot_slice_invalid__string_x_params_fail(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x','0',"aa",'1')
        y = Axis('y','2','8','3')
        intensity_start = '7'
        intensity_end = '8'
        norm_to_one = '9'
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap

        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(not self.slice_plotter.plot_slice.called)
        self.slice_view.error_invalid_x_params.assert_called_once_with()

    def test_plot_slice_x_start_bigger_than_x_stop_fail(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x','2','1','.1')
        y = Axis('y','2','8','3')
        intensity_start = '7'
        intensity_end = '8'
        norm_to_one = '9'
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap

        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(not self.slice_plotter.plot_slice.called)
        self.slice_view.error_invalid_x_params.assert_called_once_with()

    def test_plot_slice_invalid_string_y_params_fail(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x','0','7','1')
        y = Axis('y','2',"8.-",'3')
        intensity_start = '7'
        intensity_end = '8'
        norm_to_one = '9'
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap

        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(not self.slice_plotter.plot_slice.called)
        self.slice_view.error_invalid_y_params.assert_called_once_with()

    def test_plot_slice_invalid_y_params_y_end_less_than_y_start_fail(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x','0','7','1')
        y = Axis('y','20','8','3')
        intensity_start = '7'
        intensity_end = '8'
        norm_to_one = '9'
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap

        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(not self.slice_plotter.plot_slice.called)
        self.slice_view.error_invalid_y_params.assert_called_once_with()

    def test_plot_slice_invalid_intensity_params_fail(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x','0','7','1')
        y = Axis('y','2','8','3')
        intensity_start = '7'
        intensity_end = "j"
        norm_to_one = '9'
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap

        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(not self.slice_plotter.plot_slice.called)
        self.slice_view.error_invalid_intensity_params.assert_called_once_with()

    def test_plot_slice_intensity_end_less_than_intensity_start_fail(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x','0','7','1')
        y = Axis('y','2','8','3')
        intensity_start = '7'
        intensity_end = '1'
        norm_to_one = '9'
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap

        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(not self.slice_plotter.plot_slice.called)
        self.slice_view.error_invalid_intensity_params.assert_called_once_with()

    def test_plot_slice_error_handling(self):
        self.slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        self.slice_plotter_presenter.register_master(self.main_presenter)
        self.slice_plotter_presenter.register_master(self.main_presenter)
        self.slice_plotter_presenter.register_master(self.main_presenter)
        x = Axis('x', '0', '10' ,'1')
        y = Axis('y', '2', '8', '3')
        intensity_start = '7'
        intensity_end = '8'
        norm_to_one = False
        smoothing = '10'
        colourmap = 'colormap'
        selected_workspace = 'workspace1'
        self.slice_view.get_slice_x_axis.return_value = x.units
        self.slice_view.get_slice_x_start.return_value = x.start
        self.slice_view.get_slice_x_end.return_value = x.end
        self.slice_view.get_slice_x_step.return_value = x.step
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_y_start.return_value = y.start
        self.slice_view.get_slice_y_end.return_value = y.end
        self.slice_view.get_slice_y_step.return_value = y.step
        self.slice_view.get_slice_intensity_start.return_value = intensity_start
        self.slice_view.get_slice_intensity_end.return_value = intensity_end
        self.slice_view.get_slice_is_norm_to_one.return_value = norm_to_one
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_view.get_slice_colourmap.return_value = colourmap
        plot_info = ("plot_data", "boundaries", "colormap", "norm")
        self.slice_plotter.plot_slice = mock.Mock( return_value=plot_info )

        # Test empty workspace, multiple workspaces
        self.main_presenter.get_selected_workspaces.return_value = []
        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(self.slice_view.error_select_one_workspace.called)
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace, selected_workspace]
        self.slice_view.error_select_one_workspace.reset_mock()
        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(self.slice_view.error_select_one_workspace.called)
        # Test invalid axes
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_y_axis.return_value = x.units
        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(self.slice_view.error_invalid_plot_parameters.called)
        # Test invalid smoothing parameters
        self.slice_view.get_slice_y_axis.return_value = y.units
        self.slice_view.get_slice_smoothing.return_value = 'a'
        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(self.slice_view.error_invalid_smoothing_params.called)
        # Simulate matplotlib error
        self.slice_view.get_slice_smoothing.return_value = smoothing
        self.slice_plotter.plot_slice = mock.Mock(side_effect=ValueError('minvalue must be less than or equal to maxvalue'))
        self.slice_plotter_presenter.notify(Command.DisplaySlice)
        assert(self.slice_view.error_invalid_intensity_params.called)
        self.slice_plotter.plot_slice = mock.Mock(side_effect=ValueError('something bad'))
        self.assertRaises(ValueError, self.slice_plotter_presenter.notify, Command.DisplaySlice)

    def test_workspace_selection_changed_multiple_selected_empty_options_success(self):
        slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        slice_plotter_presenter.register_master(self.main_presenter)
        workspace = "a"
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace,workspace])
        axis = ["x","y"]
        self.slice_plotter.get_available_axis = mock.Mock(return_value=axis)

        slice_plotter_presenter.workspace_selection_changed()
        assert(self.slice_view.clear_input_fields.called)

    @mock.patch('mslice.presenters.slice_plotter_presenter.get_workspace_handle')
    @mock.patch('mslice.models.alg_workspace_ops.get_workspace_handle')
    @mock.patch('mslice.presenters.slice_plotter_presenter.get_available_axis')
    @mock.patch('mslice.presenters.slice_plotter_presenter.get_axis_range')
    def test_workspace_selection_changed(self, get_axis_range_mock, get_available_axis_mock, get_ws_handle_mock,
                                         get_ws_handle_mock2):
        slice_plotter_presenter = SlicePlotterPresenter( self.slice_view, self.slice_plotter )
        slice_plotter_presenter.register_master(self.main_presenter)
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        ws_mock = mock.Mock()
        get_ws_handle_mock.return_value = ws_mock
        get_ws_handle_mock2.return_value = ws_mock
        dims = ['dim1', 'dim2']
        get_available_axis_mock.return_value=dims
        get_axis_range_mock.return_value = (0, 1, 0.1)
        slice_plotter_presenter.workspace_selection_changed()
        assert(self.slice_view.populate_slice_x_options.called)
        assert(self.slice_view.populate_slice_y_options.called)
        assert(get_available_axis_mock.called)
        assert(get_axis_range_mock.called)
        # Test error handling
        get_axis_range_mock.side_effect = KeyError
        slice_plotter_presenter.workspace_selection_changed()
        assert(self.slice_view.clear_input_fields.called)

    def test_notify_presenter_clears_error(self):
        presenter = SlicePlotterPresenter(self.slice_view, self.slice_plotter)
        presenter.register_master(self.main_presenter)
        self.slice_view.clear_displayed_error = mock.Mock()
        # This unit test will verify that notifying cut presenter will cause the error to be cleared on the view.
        # The actual subsequent procedure will fail, however this irrelevant to this. Hence the try, except blocks
        for command in [x for x in dir(Command) if x[0] != "_"]:
            try:
                presenter.notify(command)
            except ValueError:
                pass
            self.slice_view.clear_displayed_error.assert_called()
            self.slice_view.reset_mock()
