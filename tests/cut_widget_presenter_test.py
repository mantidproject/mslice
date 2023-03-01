from __future__ import (absolute_import, division, print_function)
import mock
from mock import call, patch
import unittest
import warnings

from six import string_types

from mslice.models.axis import Axis
from mslice.models.alg_workspace_ops import get_available_axes
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.cut_widget_presenter import CutWidgetPresenter
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.widgets.cut.command import Command
from mslice.views.interfaces.cut_view import CutView


class CutWidgetPresenterTest(unittest.TestCase):

    def setUp(self):
        self.view = mock.create_autospec(CutView)
        self.cut_plotter_presenter = mock.create_autospec(CutPlotterPresenter)
        self.main_presenter = mock.create_autospec(MainPresenterInterface)

    def _create_cut(self, *args):
        axis, processed_axis = tuple(args[0:2])
        integration_start, integration_end, width = tuple(args[2:5])
        intensity_start, intensity_end, is_norm = tuple(args[5:8])
        workspace, integrated_axis, cut_algorithm = tuple(args[8:11])
        if isinstance(workspace, string_types):
            workspace = [workspace]
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=workspace)
        self.view.get_cut_axis = mock.Mock(return_value=axis.units)
        self.view.get_cut_axis_start = mock.Mock(return_value=axis.start)
        self.view.get_cut_axis_end = mock.Mock(return_value=axis.end)
        self.view.get_cut_axis.step = mock.Mock(return_value=axis.step)
        self.view.get_integration_axis = mock.Mock(return_value=integrated_axis)
        self.view.get_integration_start = mock.Mock(return_value=integration_start)
        self.view.get_integration_end = mock.Mock(return_value=integration_end)
        self.view.get_intensity_start = mock.Mock(return_value=intensity_start)
        self.view.get_intensity_end = mock.Mock(return_value=intensity_end)
        self.view.get_intensity_is_norm_to_one = mock.Mock(return_value=is_norm)
        self.view.get_integration_width = mock.Mock(return_value=width)
        self.view.get_energy_units = mock.Mock(return_value=axis.e_unit)
        self.view.get_cut_algorithm = mock.Mock(return_value=cut_algorithm)

    @staticmethod
    def _get_workspace_handle_method(input_workspace):
        return input_workspace

    def test_constructor_success(self):
        self.view.disable = mock.Mock()
        CutWidgetPresenter(self.view)
        self.view.disable.assert_called()

    def test_register_master_success(self):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        self.main_presenter.subscribe_to_workspace_selection_monitor.assert_called_with(cut_widget_presenter)

    def test_workspace_selection_changed_multiple_workspaces(self):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        self.main_presenter.get_selected_workspace = mock.Mock(return_value=['a', 'b'])
        for attribute in dir(CutView):
            if not attribute.startswith("__"):
                setattr(self.view, attribute, mock.Mock())
        cut_widget_presenter.workspace_selection_changed()
        # make sure only the attributes in the tuple were called and nothing else
        for attribute in dir(CutView):
            if not attribute.startswith("__"):
                if attribute in ("clear_input_fields", "disable"):
                    getattr(self.view, attribute).assert_called()
                else:
                    getattr(self.view, attribute).assert_not_called()

    def test_notify_presenter_clears_error(self):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        self.view.clear_displayed_error = mock.Mock()
        for command in [x for x in dir(Command) if x[0] != "_"]:
            cut_widget_presenter.notify(command)
            self.view.clear_displayed_error.assert_called()
            self.view.reset_mock()

    @patch('mslice.presenters.cut_widget_presenter.is_cuttable')
    @patch('mslice.presenters.cut_widget_presenter.get_workspace_handle')
    @patch('mslice.models.alg_workspace_ops.get_workspace_handle')
    def test_workspace_selection_changed_single_cuttable_workspace(self, get_ws_handle_mock2, get_ws_handle_mock,
                                                                   is_cuttable_mock):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        is_cuttable_mock.return_value=True

        ws_mock = mock.Mock()
        ws_mock.is_PSD = False
        ws_mock.limits = {}
        ws_mock.get_saved_cut_parameters = mock.Mock(return_value=(None, None))

        get_ws_handle_mock.return_value = ws_mock
        get_ws_handle_mock2.return_value = ws_mock
        available_dimensions = get_available_axes(workspace)
        cut_widget_presenter.workspace_selection_changed()
        self.view.populate_cut_axis_options.assert_called_with(available_dimensions)
        self.view.enable.assert_called_with()
        # Change workspace again, to check if cut parameters properly saved
        new_workspace = 'new_workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[new_workspace])
        fields = dict()
        fields['axes'] = available_dimensions
        self.view.get_input_fields = mock.Mock(return_value=fields)
        self.view.is_fields_cleared = mock.Mock(return_value=False)
        ws_mock.get_saved_cut_parameters.return_value=(fields, available_dimensions[0])
        ws_mock.is_axis_saved = mock.Mock(return_value=False)
        self.view.get_cut_axis = mock.Mock(return_value=available_dimensions[0])
        cut_widget_presenter.workspace_selection_changed()
        ws_mock.set_saved_cut_parameters.assert_called_with(available_dimensions[0], fields)
        self.view.get_cut_axis.assert_called_with()
        # Change back to check that it repopulates the fields
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        cut_widget_presenter.workspace_selection_changed()
        self.view.populate_input_fields.assert_called_with(fields)
        ws_mock.set_saved_cut_parameters.assert_called_with(available_dimensions[0], fields)

    @patch('mslice.presenters.cut_widget_presenter.is_cuttable')
    def test_workspace_selection_changed_single_noncut_workspace(self, is_cuttable_mock):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        is_cuttable_mock.return_value = False
        cut_widget_presenter.workspace_selection_changed()
        self.view.clear_input_fields.assert_called_with()
        self.view.disable.assert_called_with()

    def test_cut_no_workspaces_selected_fail(self):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        cut_plotter_presenter = CutPlotterPresenter()
        run_cut = mock.Mock()
        cut_plotter_presenter.run_cut = run_cut
        cut_widget_presenter.set_cut_plotter_presenter(cut_plotter_presenter)
        cut_widget_presenter.notify(Command.Plot)
        self.assertEqual([], run_cut.call_args_list)


    @patch('mslice.presenters.cut_widget_presenter.is_cuttable')
    @patch('mslice.presenters.cut_widget_presenter.get_workspace_handle')
    @patch('mslice.models.alg_workspace_ops.get_workspace_handle')
    def test_change_axis(self, get_ws_handle_mock, get_ws_handle_mock2, is_cuttable_mock):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        # Set up a mock workspace with two sets of cutable axes, then change to this ws
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        is_cuttable_mock.return_value = True
        ws_mock = mock.Mock()
        ws_mock.is_PSD = False
        ws_mock.limits = {}
        ws_mock.get_saved_cut_parameters = mock.Mock(return_value=(None, None))
        get_ws_handle_mock.return_value = ws_mock
        get_ws_handle_mock2.return_value = ws_mock
        cut_widget_presenter.workspace_selection_changed()
        # Set up a set of input values for this cut, then simulate changing axes.
        fields1 = dict()
        fields1['axes'] = '|Q|'
        fields1['cut_parameters'] = ['0', '10', '0.05']
        fields1['integration_range'] = ['-1', '1']
        fields1['integration_width'] = '2'
        fields1['normtounity'] = False
        self.view.get_input_fields = mock.Mock(return_value=fields1)
        self.view.get_cut_axis = mock.Mock(return_value='DeltaE')
        self.view.is_fields_cleared = mock.Mock(return_value=False)
        self.view.populate_input_fields = mock.Mock()
        cut_widget_presenter.notify(Command.AxisChanged)
        ws_mock.set_saved_cut_parameters.assert_called_with('|Q|', fields1)
        self.view.clear_input_fields.assert_called_with(keep_axes=True)
        self.view.populate_input_fields.assert_not_called()
        # Set up a set of input values for this other cut, then simulate changing axes again.
        fields2 = dict()
        fields2['axes'] = 'DeltaE'
        fields2['cut_parameters'] = ['-5', '5', '0.1']
        fields2['integration_range'] = ['2', '3']
        fields2['integration_width'] = '1'
        fields2['normtounity'] = True
        self.view.get_input_fields = mock.Mock(return_value=fields2)
        self.view.get_cut_axis = mock.Mock(return_value='|Q|')
        ws_mock.get_saved_cut_parameters = mock.Mock(return_value=(fields1, '|Q|'))
        cut_widget_presenter.notify(Command.AxisChanged)
        ws_mock.set_saved_cut_parameters.assert_called_with('DeltaE', fields2)
        ws_mock.get_saved_cut_parameters.assert_called_with('|Q|')
        self.view.populate_input_fields.assert_called_with(fields1)

    @patch('mslice.presenters.cut_widget_presenter.is_cuttable')
    @patch('mslice.presenters.cut_widget_presenter.get_axis_range')
    @patch('mslice.presenters.cut_widget_presenter.get_workspace_handle')
    @patch('mslice.models.alg_workspace_ops.get_workspace_handle')
    def test_cut_step_size(self, get_ws_handle_mock, get_ws_handle_mock2, get_axis_range_mock, is_cuttable_mock):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        is_cuttable_mock.return_value = True
        ws_mock = mock.Mock()
        ws_mock.is_PSD = False
        ws_mock.limits = {}
        ws_mock.get_saved_cut_parameters = mock.Mock(return_value=(None, None))
        get_ws_handle_mock.return_value = ws_mock
        get_ws_handle_mock2.return_value = ws_mock
        cut_widget_presenter.workspace_selection_changed()
        get_axis_range_mock.assert_any_call(ws_mock, '|Q|')
        get_axis_range_mock.assert_any_call(ws_mock, 'DeltaE')
        get_axis_range_mock.side_effect = KeyError
        cut_widget_presenter.workspace_selection_changed()
        self.view.set_minimum_step.assert_called_with(None)

    def test_invalid_step(self):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        axis = Axis("units", "0", "100", 0)
        processed_axis = Axis("units", 0, 100, 0)
        integration_start = 3
        integration_end = 5
        width = ""
        intensity_start = 11
        intensity_end = 30
        is_norm = True
        workspace = "workspace"
        integrated_axis = 'integrated axis'
        cut_algorithm = 0
        self._create_cut(axis, processed_axis, integration_start, integration_end, width,
                         intensity_start, intensity_end, is_norm, workspace, integrated_axis,
                         cut_algorithm)
        self.view.get_cut_axis_step = mock.Mock(return_value="")
        self.view.get_minimum_step = mock.Mock(return_value=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cut_widget_presenter.notify(Command.Plot)
        self.view.get_minimum_step.assert_called_with()
        self.view.display_error.assert_any_call('Invalid cut step parameter, using default.')
        self.view.populate_cut_params.assert_called_with(0.0, 100.0, '1.00000')

    @patch('mslice.presenters.cut_widget_presenter.get_workspace_handle')
    def test_plot_multiple_workspaces_cut(self, mock_get_workspace_handle):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        cut_plotter_presenter = mock.MagicMock()
        cut_widget_presenter.set_cut_plotter_presenter(cut_plotter_presenter)
        axis = Axis("units", "0", "100", "1")
        integration_start = 3
        integration_end = 8
        width = ""
        intensity_start = 11
        intensity_end = 30
        is_norm = True
        ws1 = mock.MagicMock()
        ws1.return_value.return_value.e_fixed = 1
        ws2 = mock.MagicMock()
        ws2.return_value.return_value.e_fixed = 1
        selected_workspaces = [ws1, ws2]
        integrated_axis = 'integrated axis'
        cut_algorithm = 0
        integration_axis = Axis('integrated axis', integration_start, integration_end, 0)
        mock_get_workspace_handle.side_effect = self._get_workspace_handle_method

        self._create_cut(axis, integration_axis, integration_start, integration_end, width,
                         intensity_start, intensity_end, is_norm, selected_workspaces, integrated_axis,
                         cut_algorithm)
        cut_widget_presenter.notify(Command.Plot)
        call_list = [
            call(selected_workspaces[0], mock.ANY, save_only=False, plot_over=False),
            call(selected_workspaces[1], mock.ANY, save_only=False, plot_over=True),
        ]
        cut_plotter_presenter.run_cut.assert_has_calls(call_list)

    def test_set_energy_default(self):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.set_energy_default("meV")
        self.view.set_energy_units.assert_called_once()
        self.view.set_energy_units_default.assert_called_once()

    @patch('mslice.presenters.cut_widget_presenter.get_workspace_handle')
    @patch('mslice.presenters.cut_widget_presenter.Cut')
    def test_cut_integration_algorithm(self, mock_cut_obj, mock_get_workspace_handle):
        cut_widget_presenter = CutWidgetPresenter(self.view)
        cut_widget_presenter.register_master(self.main_presenter)
        cut_plotter_presenter = mock.MagicMock()
        cut_plotter_presenter.run_cut = mock.Mock()
        cut_widget_presenter.set_cut_plotter_presenter(cut_plotter_presenter)
        integrated_axis, integration_start, integration_end, width = ('integrated axis', 3, 8, "")
        intensity_start, intensity_end, is_norm, workspace = (11, 30, True, 'ws1')
        axis = Axis("units", "0", "100", "1")
        integration_axis = Axis('integrated axis', integration_start, integration_end, 0)
        cut_algorithm = 1
        mock_get_workspace_handle.return_value.e_fixed = 1

        self._create_cut(axis, integration_axis, integration_start, integration_end, width,
                         intensity_start, intensity_end, is_norm, workspace, integrated_axis,
                         cut_algorithm)
        cut_widget_presenter.notify(Command.Plot)
        self.view.display_error.assert_not_called()
        mock_cut_obj.assert_called_once_with(axis, integration_axis, intensity_start, intensity_end,
                                             is_norm, width, 'Integration', None, 1)
