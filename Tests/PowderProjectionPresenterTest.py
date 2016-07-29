import unittest
import mock
from mock import call

from powderprojection.PowderProjectionPresenter import PowderProjectionPresenter
from powderprojection.PowderProjectionView import PowderView
from powderprojection.ProjectionCalculator import ProjectionCalculator
from powderprojection.command import Command
from mainview import MainView
from MainPresenter import MainPresenter


class PowderProjectionPresenterTest(unittest.TestCase):
    def setUp(self):
        # Set up a mock view, presenter, main view and main presenter
        self.powder_view = mock.create_autospec(PowderView)
        self.projection_calculator = mock.create_autospec(ProjectionCalculator)
        self.main_presenter = mock.create_autospec(MainPresenter)
        self.mainview = mock.create_autospec(MainView)
        self.mainview.get_presenter = mock.Mock(return_value=self.main_presenter)

    def test_constructor_success(self):
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.mainview, self.projection_calculator)

    def test_constructor_incorrect_powder_view_fail(self):
        self.assertRaises(TypeError, PowderProjectionPresenter, self.mainview, self.mainview, self.projection_calculator)

    def test_constructor_incorrect_main_view_fail(self):
        self.assertRaises(TypeError, PowderProjectionPresenter, self.powder_view, self.powder_view, self.projection_calculator)

    def test_constructor_incorrect_projection_calculator_fail(self):
        self.assertRaises(TypeError, PowderProjectionPresenter, self.powder_view, self.mainview, None)

    def test_calculate_projection_success(self):
        selected_workspace = 'a'
        output_workspace = 'b'
        # Setting up view to report output_workspace as output workspace name supplied by user
        self.powder_view.get_output_workspace_name = mock.Mock(return_value=output_workspace)
        # Setting up main presenter to report that the current selected workspace is selected_workspace
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[selected_workspace])
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.mainview, self.projection_calculator)
        # Setup view to report Energy and |Q| as selected axis to project two
        u1 = 'Energy'
        u2 = '|Q|'
        self.powder_view.get_powder_u1 = mock.Mock(return_value=u1)
        self.powder_view.get_powder_u2 = mock.Mock(return_value=u2)
        self.powder_presenter.notify(Command.CalculatePowderProjection)
        self.main_presenter.get_selected_workspaces.assert_called_once_with()
        self.powder_view.get_powder_u1.assert_called_once_with()
        self.powder_view.get_powder_u2.assert_called_once_with()
        self.powder_view.get_output_workspace_name.assert_called_with()
        # TODO edit after recieving binning specs (test binning recieved from user if appropriate)
        #TODO make test more strict after recieving binning specs
        #self.projection_calculator.calculate_projection.assert_called_once_with(input_workspace=selected_workspace,
        #                           output_workspace=output_workspace,qbinning=???,axis1=u1,axis2=u2)
        self.projection_calculator.calculate_projection.assert_called_once()
        self.main_presenter.update_displayed_workspaces.assert_called_once_with()

    def test_notify_presenter_with_unrecognised_command_raise_exception(self):
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.mainview, self.projection_calculator)
        unrecognised_command = 1234567
        self.assertRaises(ValueError, self.powder_presenter.notify, unrecognised_command)
