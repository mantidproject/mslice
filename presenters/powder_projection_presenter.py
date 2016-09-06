from models.projection.powder.projection_calculator import ProjectionCalculator
from views.powder_projection_view import PowderView
from widgets.projection.powder.command import Command
from validation_decorators import require_main_presenter
from presenters.interfaces.main_presenter import MainPresenterInterface
from interfaces.powder_projection_presenter import PowderProjectionPresenterInterface


class PowderProjectionPresenter(PowderProjectionPresenterInterface):

    def __init__(self, powder_view, projection_calculator):
        self._powder_view = powder_view
        self._projection_calculator = projection_calculator
        if not isinstance(self._powder_view,PowderView):
            raise TypeError("powder_view is not of type PowderView")
        if not isinstance(self._projection_calculator,ProjectionCalculator):
            raise TypeError("projection_calculator is not of type ProjectionCalculator")

        #Add rest of options
        self._available_units = ['|Q|', 'Energy']
        self._powder_view.populate_powder_projection_axis(self._available_units)
        self._main_presenter = None

    def register_master(self, main_presenter):
        assert (isinstance(main_presenter, MainPresenterInterface))
        self._main_presenter = main_presenter

    def notify(self, command):
        if command == Command.CalculatePowderProjection:
            self._calculate_powder_projection()
        elif command == Command.U1Changed:
            self._axis_changed(1)
        elif command == Command.U2Changed:
            self._axis_changed(2)
        else:
            raise ValueError("Powder Projection Presenter received an unrecognised command")

    def _calculate_powder_projection(self):
        selected_workspaces = self._get_main_presenter().get_selected_workspaces()
        axis1 = self._powder_view.get_powder_u1()
        axis2 = self._powder_view.get_powder_u2()
        if axis1 == axis2:
            raise NotImplementedError('equal axis')
        if not selected_workspaces:
            raise NotImplementedError('Implement error message')

        for workspace in selected_workspaces:
            self._projection_calculator.calculate_projection(workspace, axis1, axis2)
        self._get_main_presenter().update_displayed_workspaces()

    @require_main_presenter
    def _get_main_presenter(self):
        return self._main_presenter

    def _axis_changed(self, axis):
        """This a private method which stops U1 from being the same as u2 on the gui at any point"""
        if axis == 1:
            all_axis = self._available_units[:]
            if all_axis[0] == self._powder_view.get_powder_u1():
                all_axis = all_axis[::-1] # reverse list pushing what selected in u1 to the bottom
            self._powder_view.populate_powder_u2(all_axis)
        if axis == 2:
            all_axis = self._available_units[:]
            if all_axis[0] == self._powder_view.get_powder_u2():
                all_axis = all_axis[::-1] # reverse list pushing what selected in u1 to the bottom
            self._powder_view.populate_powder_u1(all_axis)