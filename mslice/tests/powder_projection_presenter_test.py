from __future__ import (absolute_import, division, print_function)
import mock
import unittest

from mslice.models.projection.powder.projection_calculator import ProjectionCalculator
from mslice.presenters.interfaces.main_presenter import MainPresenterInterface
from mslice.presenters.powder_projection_presenter import PowderProjectionPresenter
from mslice.views.mainview import MainView
from mslice.views.powder_projection_view import PowderView
from mslice.widgets.projection.powder.command import Command


class PowderProjectionPresenterTest(unittest.TestCase):
    def setUp(self):
        # Set up a mock view, presenter, main view and main presenter
        self.powder_view = mock.create_autospec(PowderView)
        self.projection_calculator = mock.create_autospec(ProjectionCalculator)
        self.projection_calculator.configure_mock(**{'available_axes.return_value': ['|Q|', '2Theta', 'DeltaE']})
        self.projection_calculator.configure_mock(**{'available_units.return_value': ['meV', 'cm-1']})
        self.main_presenter = mock.create_autospec(MainPresenterInterface)
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
        powder_presenter.register_master(self.main_presenter)

    def test_register_master_invalid_master_fail(self):
        powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        self.assertRaises(AssertionError, powder_presenter.register_master, 3)

    def test_calculate_projection_success(self):
        selected_workspace = 'a'
        # Setting up main presenter to report that the current selected workspace is selected_workspace
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[selected_workspace])
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        self.powder_presenter.register_master(self.main_presenter)
        # Setup view to report DeltaE and |Q| as selected axis to project two
        u1 = 'DeltaE'
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
        self.main_presenter.set_selected_workspaces.assert_called_once()

    def test_notify_presenter_with_unrecognised_command_raise_exception(self):
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        unrecognised_command = 1234567
        self.assertRaises(ValueError, self.powder_presenter.notify, unrecognised_command)

    def test_calculate_projection_equal_axis_error(self):
        selected_workspace = 'a'
        # Setting up main presenter to report that the current selected workspace is selected_workspace
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=[selected_workspace])
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        # Setup view to report DeltaE and |Q| as selected axis to project two
        u1 = 'DeltaE'
        u2 = 'DeltaE'
        self.powder_view.get_powder_u1 = mock.Mock(return_value=u1)
        self.powder_view.get_powder_u2 = mock.Mock(return_value=u2)
        self.powder_presenter.register_master(self.main_presenter)
        self.assertRaises(ValueError,self.powder_presenter.notify,Command.CalculatePowderProjection)
        self.main_presenter.get_selected_workspaces.assert_called_once_with()
        self.powder_view.get_powder_u1.assert_called_once_with()
        self.powder_view.get_powder_u2.assert_called_once_with()

        self.projection_calculator.calculate_projection.assert_not_called()

    def test_calculate_projection_multiple_selection(self):
        selected_workspaces = []
        # Setting up main presenter to report that the current selected workspace is selected_workspace
        self.main_presenter.get_selected_workspaces = mock.Mock(return_value=selected_workspaces)
        self.powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        self.powder_presenter.register_master(self.main_presenter)
        # Setup view to report DeltaE and |Q| as selected axis to project two
        u1 = 'DeltaE'
        u2 = '|Q|'
        self.powder_view.get_powder_u1 = mock.Mock(return_value=u1)
        self.powder_view.get_powder_u2 = mock.Mock(return_value=u2)
        self.assertRaises(NameError,self.powder_presenter.notify,Command.CalculatePowderProjection)
        self.main_presenter.get_selected_workspaces.assert_called_once_with()
        self.powder_view.get_powder_u1.assert_called_once_with()
        self.powder_view.get_powder_u2.assert_called_once_with()

        self.projection_calculator.calculate_projection.assert_not_called()

    def test_notify_presenter_clears_error(self):
        presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        presenter.register_master(self.main_presenter)
        # This unit test will verify that notifying cut presenter will cause the error to be cleared on the view.
        # The actual subsequent procedure will fail, however this irrelevant to this. Hence the try, except blocks
        for command in [x for x in dir(Command) if x[0] != "_"]:
            try:
                presenter.notify(command)
            except ValueError:
                pass
            self.powder_view.clear_displayed_error.assert_called()
            self.powder_view.reset_mock()

    def test_axis_switching_1(self):
        powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        dropbox_contents = self.powder_view.populate_powder_u1.call_args[0][0]
        # Makes u2 == u1
        self.powder_view.set_powder_u1.reset_mock()
        self.powder_view.set_powder_u2.reset_mock()
        self.powder_view.get_powder_u1 = mock.Mock(return_value=dropbox_contents[1])
        self.powder_view.get_powder_u2 = mock.Mock(return_value=dropbox_contents[1])
        # Now set u1 to change, and check that u1 is not subsequently affected, but u2 is called.
        powder_presenter.notify(Command.U1Changed)
        self.powder_view.set_powder_u1.assert_not_called()
        # Since we set u1 to be a non-DeltaE axis, u2 must be DeltaE
        self.powder_view.set_powder_u2.assert_called_with(dropbox_contents[-1])

    def test_axis_switching_2(self):
        powder_presenter = PowderProjectionPresenter(self.powder_view, self.projection_calculator)
        dropbox_contents = self.powder_view.populate_powder_u2.call_args[0][0]
        # Makes u2 == u1 == DeltaE
        self.powder_view.set_powder_u1.reset_mock()
        self.powder_view.set_powder_u2.reset_mock()
        self.powder_view.get_powder_u1 = mock.Mock(return_value=dropbox_contents[2])
        self.powder_view.get_powder_u2 = mock.Mock(return_value=dropbox_contents[2])
        # Now set u2 to change, and check that u2 is not subsequently affected, but u1 is called.
        powder_presenter.notify(Command.U2Changed)
        self.powder_view.set_powder_u2.assert_not_called()
        # Since we set u1==u2==DeltaE, u1 must now be the first non-DeltaE unit
        self.powder_view.set_powder_u1.assert_called_once_with(dropbox_contents[0])
