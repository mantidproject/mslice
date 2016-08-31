import unittest
import mock
from mock import call
from presenters.cut_presenter import CutPresenter
from views.cut_view import CutView
from models.cut.cut_algorithm import CutAlgorithm
from presenters.interfaces.main_presenter import MainPresenterInterface

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

    def test_workspace_selection_changed_single_cut_workspace(self):
        cut_presenter = CutPresenter(self.view, self.cut_algorithm, self.plotting_module)
        cut_presenter.register_master(self.main_presenter)
        workspace = 'workspace'
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[workspace])
        self.cut_algorithm.is_cuttable = mock.Mock(return_value=False)
        self.cut_algorithm.is_cut = mock.Mock(return_value=True)
        available_dimensions = ["dim1", "dim2"]
        self.cut_algorithm.get_available_axis = mock.Mock(return_value=available_dimensions)

        cut_presenter.workspace_selection_changed()
        self.view.populate_cut_axis_options.assert_called_with(available_dimensions)
        self.view.enable.assert_called_with()
