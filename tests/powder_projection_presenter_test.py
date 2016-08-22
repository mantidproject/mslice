import unittest

import mock

from mainview import MainView
from models.projection.powder.projection_calculator import ProjectionCalculator
from presenters.main_presenter import MainPresenter
from presenters.powder_projection_presenter import PowderProjectionPresenter
from views.powder_projection_view import PowderView
from widgets.projection.powder.command import Command


class PowderProjectionPresenterTest(unittest.TestCase):
    def setUp(self):
        # Set up a mock view, presenter, main view and main presenter
        self.powder_view = mock.create_autospec(PowderView)
        self.projection_calculator = mock.create_autospec(ProjectionCalculator)
        self.main_presenter = mock.create_autospec(MainPresenter)
        self.mainview = mock.create_autospec(MainView)
        self.mainview.get_presenter = mock.Mock(return_value=self.main_presenter)

    def test_constructor_success(self):
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)

    def test_constructor_incorrect_powder_view_fail(self):
        self.assertRaises(TypeError, PowderProjectionPresenter, self.mainview, self.mainview, self.projection_calculator)

    def test_constructor_incorrect_main_view_fail(self):
        self.assertRaises(TypeError, PowderProjectionPresenter, self.powder_view, self.powder_view, self.projection_calculator)

    def test_constructor_incorrect_projection_calculator_fail(self):
        self.assertRaises(TypeError, PowderProjectionPresenter, self.powder_view, self.mainview, None)

    def test_register_master(self):
        powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        powder_presenter.register_master(self.mainview)

    def test_register_master_invalid_master_fail(self):
        powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        self.assertRaises(AssertionError, powder_presenter.register_master, 3)

    def test_calculate_projection_success(self):
        selected_workspace = 'a'
        output_workspace = 'b'
        # Setting up main presenter to report that the current selected workspace is selected_workspace
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[selected_workspace])
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        self.powder_presenter.register_master(self.mainview)
        # Setup view to report Energy and |Q| as selected axis to project two
        u1 = 'Energy'
        u2 = '|Q|'
        self.powder_view.get_powder_u1 = mock.Mock(return_value=u1)
        self.powder_view.get_powder_u2 = mock.Mock(return_value=u2)
        self.powder_presenter.notify(Command.CalculatePowderProjection)
        self.main_presenter.get_selected_workspaces.assert_called_once_with()
        self.powder_view.get_powder_u1.assert_called_once_with()
        self.powder_view.get_powder_u2.assert_called_once_with()
        # TODO edit after recieving binning specs (test binning recieved from user if appropriate)
        #TODO make test more strict after recieving binning specs
        #self.projection_calculator.calculate_projection.assert_called_once_with(input_workspace=selected_workspace,
        #                           output_workspace=output_workspace,qbinning=???,axis1=u1,axis2=u2)
        self.projection_calculator.calculate_projection.assert_called_once()
        self.main_presenter.update_displayed_workspaces.assert_called_once_with()

    def test_notify_presenter_with_unrecognised_command_raise_exception(self):
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        unrecognised_command = 1234567
        self.assertRaises(ValueError, self.powder_presenter.notify, unrecognised_command)

    def test_calculate_projection_equal_axis_error(self):
        selected_workspace = 'a'
        output_workspace = 'b'
        # Setting up main presenter to report that the current selected workspace is selected_workspace
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[selected_workspace])
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        # Setup view to report Energy and |Q| as selected axis to project two
        u1 = 'Energy'
        u2 = 'Energy'
        self.powder_view.get_powder_u1 = mock.Mock(return_value=u1)
        self.powder_view.get_powder_u2 = mock.Mock(return_value=u2)
        self.powder_presenter.register_master(self.mainview)
        self.assertRaises(NotImplementedError,self.powder_presenter.notify,Command.CalculatePowderProjection)
        self.main_presenter.get_selected_workspaces.assert_called_once_with()
        self.powder_view.get_powder_u1.assert_called_once_with()
        self.powder_view.get_powder_u2.assert_called_once_with()

        self.projection_calculator.calculate_projection.assert_not_called()

    def test_calculate_projection_multiple_selection(self):
        selected_workspaces = []
        # Setting up main presenter to report that the current selected workspace is selected_workspace
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=selected_workspaces)
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        self.powder_presenter.register_master(self.mainview)
        # Setup view to report Energy and |Q| as selected axis to project two
        u1 = 'Energy'
        u2 = '|Q|'
        self.powder_view.get_powder_u1 = mock.Mock(return_value=u1)
        self.powder_view.get_powder_u2 = mock.Mock(return_value=u2)
        self.assertRaises(NotImplementedError,self.powder_presenter.notify,Command.CalculatePowderProjection)
        self.main_presenter.get_selected_workspaces.assert_called_once_with()
        self.powder_view.get_powder_u1.assert_called_once_with()
        self.powder_view.get_powder_u2.assert_called_once_with()

        self.projection_calculator.calculate_projection.assert_not_called()
