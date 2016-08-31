import unittest
import mock
from mock import call
from presenters.cut_presenter import CutPresenter
from views.cut_view import CutView
from models.cut.cut_algorithm import CutAlgorithm
from presenters.interfaces.main_presenter import MainPresenterInterface
from presenters.slice_plotter_presenter import Axis
from widgets.cut.command import Command

class CutPresenterTest(unittest.TestCase):
    def setUp(self):
        self.view = mock.create_autospec(CutView)
        self.cut_algorithm = mock.Mock(CutAlgorithm)
        self.plotting_module = mock.Mock(spec=['errorbar', 'legend', 'xlabel', 'ylabel', 'autoscale', 'ylim'])
        self.main_presenter = mock.create_autospec(MainPresenterInterface)

    def test_constructor_success(self):
        cut_presenter = CutPresenter(self.view, self.cut_algorithm, self.plotting_module)
        self.view.disable.assert_called()

    def test_register_master_success(self):
        cut_presenter = CutPresenter(self.view, self.cut_algorithm, self.plotting_module)
        cut_presenter.register_master(self.main_presenter)
        self.main_presenter.subscribe_to_workspace_selection_monitor.assert_called_with(cut_presenter)

    def test_workspace_selection_changed_multiple_workspaces(self):
        cut_presenter = CutPresenter(self.view, self.cut_algorithm, self.plotting_module)
        cut_presenter.register_master(self.main_presenter)
        self.main_presenter.get_selected_workspace = mock.Mock(return_value=['a', 'b'])

        cut_presenter.workspace_selection_changed()
        # make sure only the attributes in the tuple were called and nothing else
        for attribute in dir(CutView):
            if not attribute.startswith("__"):
                if attribute in ("clear_input_fields", "disable"):
                    getattr(self.view, attribute).assert_called()
                else:
                    getattr(self.view, attribute).assert_not_called()


    def test_workspace_selection_changed_single_cuttable_workspace(self):
        cut_presenter = CutPresenter(self.view, self.cut_algorithm, self.plotting_module)
        cut_presenter.register_master(self.main_presenter)
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        self.cut_algorithm.is_cuttable = mock.Mock(return_value=True)
        self.cut_algorithm.is_cut = mock.Mock(return_value=False)
        available_dimensions = ["dim1", "dim2"]
        self.cut_algorithm.get_available_axis = mock.Mock(return_value=available_dimensions)

        cut_presenter.workspace_selection_changed()
        self.view.populate_cut_axis_options.assert_called_with(available_dimensions)
        self.view.enable.assert_called_with()

    # def test_workspace_selection_changed_single_cut_workspace(self):
    #     cut_presenter = CutPresenter(self.view, self.cut_algorithm, self.plotting_module)
    #     cut_presenter.register_master(self.main_presenter)
    #     workspace = 'workspace'
    #     self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
    #     self.cut_algorithm.is_cuttable = mock.Mock(return_value=False)
    #     self.cut_algorithm.is_cut = mock.Mock(return_value=True)
    #     available_dimensions = ["dim1", "dim2"]
    #     self.cut_algorithm.get_cut_params
    #     cut_presenter.workspace_selection_changed()
    #     self.view.populate_cut_axis_options.assert_called_with(available_dimensions)
    #     self.view.enable.assert_called_with()

    def test_plot_single_cut_success(self):
        cut_presenter = CutPresenter(self.view, self.cut_algorithm, self.plotting_module)
        cut_presenter.register_master(self.main_presenter)
        axis = Axis("units", "0", "100", "1")
        processed_axis = Axis("units", 0, 100, 1)
        integration_start = 3
        integration_end = 5
        width = ""
        intensity_start = 11
        intensity_end = 30
        is_norm = True
        workspace = "workspace"
        integrated_axis = 'integrated axis'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        self.view.get_cut_axis = mock.Mock(return_value=axis.units)
        self.view.get_cut_axis_start = mock.Mock(return_value=axis.start)
        self.view.get_cut_axis_end = mock.Mock(return_value=axis.end)
        self.view.get_cut_axis.step = mock.Mock(return_value=axis.step)
        self.view.get_integration_start = mock.Mock(return_value=integration_start)
        self.view.get_integration_end = mock.Mock(return_value=integration_end)
        self.view.get_intensity_start = mock.Mock(return_value=intensity_start)
        self.view.get_intensity_end = mock.Mock(return_value=intensity_end)
        self.view.get_intensity_is_norm_to_one = mock.Mock(return_value=is_norm)
        self.view.get_integration_width = mock.Mock(return_value=width)
        self.cut_algorithm.compute_cut_xye = mock.Mock(return_value=('x', 'y', 'e'))
        self.cut_algorithm.get_other_axis = mock.Mock(return_value=integrated_axis)
        cut_presenter.notify(Command.Plot)
        assert isinstance(self.cut_algorithm, CutAlgorithm)
        self.cut_algorithm.compute_cut_xye.assert_called_with(workspace, processed_axis, integration_start,
                                                              integration_end, is_norm)
