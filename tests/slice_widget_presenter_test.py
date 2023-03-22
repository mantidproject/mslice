import mock
import unittest

from mslice.models.axis import Axis
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.presenters.slice_widget_presenter import SliceWidgetPresenter
from mslice.views.interfaces.slice_view import SliceView
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.widgets.slice.command import Command


class SliceWidgetPresenterTest(unittest.TestCase):

    def setUp(self):
        self.slice_plotter_presenter = mock.create_autospec(SlicePlotterPresenter)
        self.slice_view = mock.create_autospec(SliceView)
        self.main_presenter = mock.create_autospec(MainPresenterInterface)

    def test_constructor_success(self):
        self.slice_plotter_presenter = SliceWidgetPresenter(self.slice_view)

    def test_constructor_invalid_slice_view_failure(self):
        with self.assertRaises(TypeError):
            SliceWidgetPresenter(self.slice_plotter_presenter)

    def test_notify_presenter_unknown_command_raise_exception_failure(self):
        slice_widget_presenter = SliceWidgetPresenter(self.slice_view)
        slice_widget_presenter.register_master(self.main_presenter)
        unknown_command = -1
        with self.assertRaises(ValueError):
            slice_widget_presenter.notify(unknown_command)

    def test_register_master_success(self):
        slice_presenter = SliceWidgetPresenter(self.slice_view)
        slice_presenter.register_master(self.main_presenter)
        self.main_presenter.subscribe_to_workspace_selection_monitor.assert_called_once_with(slice_presenter)

    def test_register_master_invalid_master_fail(self):
        slice_presenter = SliceWidgetPresenter(self.slice_view)
        with self.assertRaises(AssertionError):
            slice_presenter.register_master(3)

    def test_plot_slice_successful(self):
        slice_widget_presenter = SliceWidgetPresenter(self.slice_view)
        slice_widget_presenter.register_master(self.main_presenter)
        slice_widget_presenter.set_slice_plotter_presenter(self.slice_plotter_presenter)
        x = Axis('x', '0', '10' ,'1')
        y = Axis('y', '2', '8', '3')
        intensity_start = '7'
        intensity_end = '8'
        norm_to_one = False
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
        self.slice_view.get_slice_colourmap.return_value = colourmap
        self.slice_view.get_units.return_value = 'meV'
        plot_info = ("plot_data", "boundaries", "colormap", "norm")
        self.slice_plotter_presenter.plot_slice = mock.Mock(return_value=plot_info)
        self.slice_plotter_presenter.validate_intensity = mock.Mock(return_value=(7.0, 8.0))
        slice_widget_presenter.notify(Command.DisplaySlice)

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
        self.slice_view.get_slice_colourmap.assert_called_once_with()
        self.slice_plotter_presenter.validate_intensity.assert_called_once_with('7', '8')
        self.slice_plotter_presenter.plot_slice.assert_called_with(selected_workspace, Axis('x', 0, 10, 1),
                                                                   Axis('y', 2, 8, 3), float(intensity_start),
                                                                   float(intensity_end), norm_to_one, colourmap)

    def test_plot_slice_error_handling(self):
        slice_widget_presenter = SliceWidgetPresenter(self.slice_view)
        slice_widget_presenter.register_master(self.main_presenter)
        slice_widget_presenter.set_slice_plotter_presenter(self.slice_plotter_presenter)
        x = Axis('x', '0', '10', '1')
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
        self.slice_view.get_units.return_value = 'meV'
        plot_info = ("plot_data", "boundaries", "colormap", "norm")
        self.slice_plotter_presenter.plot_slice = mock.Mock(return_value=plot_info)
        self.slice_plotter_presenter.validate_intensity = mock.Mock(return_value=(7.0, 8.0))
        # Test empty workspace, multiple workspaces
        self.main_presenter.get_selected_workspaces.return_value = []
        slice_widget_presenter.notify(Command.DisplaySlice)
        assert self.slice_view.error_select_one_workspace.called
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace, selected_workspace]
        self.slice_view.error_select_one_workspace.reset_mock()
        slice_widget_presenter.notify(Command.DisplaySlice)
        assert self.slice_view.error_select_one_workspace.called
        # Test invalid axes
        self.main_presenter.get_selected_workspaces.return_value = [selected_workspace]
        self.slice_view.get_slice_y_axis.return_value = x.units
        slice_widget_presenter.notify(Command.DisplaySlice)
        assert self.slice_view.error_invalid_plot_parameters.called
        # Simulate matplotlib error
        self.slice_plotter_presenter.plot_slice = mock.Mock(
            side_effect=ValueError('minvalue must be less than or equal to maxvalue'))
        self.slice_view.get_slice_y_axis.return_value = y.units
        slice_widget_presenter.notify(Command.DisplaySlice)
        assert self.slice_view.error_invalid_intensity_params.called
        self.slice_plotter_presenter.plot_slice = mock.Mock(side_effect=ValueError('something bad'))
        self.assertRaises(ValueError, slice_widget_presenter.notify, Command.DisplaySlice)

    def test_workspace_selection_changed_multiple_selected_empty_options_success(self):
        slice_widget_presenter = SliceWidgetPresenter(self.slice_view)
        slice_widget_presenter.register_master(self.main_presenter)
        workspace = "a"
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace,workspace])
        slice_widget_presenter.workspace_selection_changed()
        assert(self.slice_view.clear_input_fields.called)

    @mock.patch('mslice.presenters.slice_widget_presenter.get_workspace_handle')
    @mock.patch('mslice.models.alg_workspace_ops.get_workspace_handle')
    @mock.patch('mslice.presenters.slice_widget_presenter.get_available_axes')
    @mock.patch('mslice.presenters.slice_widget_presenter.get_axis_range')
    @mock.patch('mslice.presenters.slice_widget_presenter.is_sliceable')
    def test_workspace_selection_changed(self, is_sliceable_mock, get_axis_range_mock, get_available_axes_mock,
                                         get_ws_handle_mock, get_ws_handle_mock2):

        slice_widget_presenter = SliceWidgetPresenter(self.slice_view)
        slice_widget_presenter.register_master(self.main_presenter)
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        ws_mock = mock.Mock()
        get_ws_handle_mock.return_value = ws_mock
        get_ws_handle_mock2.return_value = ws_mock
        is_sliceable_mock.return_value = True
        dims = ['dim1', 'dim2']
        get_available_axes_mock.return_value = dims
        get_axis_range_mock.return_value = (0, 1, 0.1)
        self.slice_view.get_units = mock.Mock(side_effect=['meV', 'cm-1', 'cm-1'])
        slice_widget_presenter.workspace_selection_changed()
        assert (self.slice_view.get_slice_x_axis.call_count == 1)
        assert (self.slice_view.populate_slice_x_options.called)
        assert (self.slice_view.populate_slice_y_options.called)
        assert (get_available_axes_mock.called)
        assert (get_axis_range_mock.called)
        # Test energy unit conversion is different for second call
        slice_widget_presenter.workspace_selection_changed()
        self.slice_view.get_slice_x_axis.assert_called()
        self.slice_view.get_slice_x_axis.return_value = 'DeltaE'
        slice_widget_presenter.workspace_selection_changed()
        assert (self.slice_view.get_slice_x_axis.call_count == 5)
        # Test error handling
        get_axis_range_mock.side_effect = KeyError
        slice_widget_presenter.workspace_selection_changed()
        assert (self.slice_view.clear_input_fields.called)

    def test_notify_presenter_clears_error(self):
        presenter = SliceWidgetPresenter(self.slice_view)
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

    def test_set_energy_default(self):
        slice_presenter = SliceWidgetPresenter(self.slice_view)
        slice_presenter.set_energy_default("meV")
        self.slice_view.set_units.assert_called_once()
        self.slice_view.set_energy_default.assert_called_once()
