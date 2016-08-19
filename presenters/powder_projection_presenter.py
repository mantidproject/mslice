from models.projection.powder.projection_calculator import ProjectionCalculator
from views.powder_projection_view import PowderView
from widgets.projection.powder.command import Command
from validation_decorators import require_main_presenter


class PowderProjectionPresenter(object):

    def __init__(self, powder_view, projection_calculator):
        self._powder_view = powder_view
        self._projection_calculator = projection_calculator
        if not isinstance(self._powder_view,PowderView):
            raise TypeError("powder_view is not of type PowderView")
        if not isinstance(self._projection_calculator,ProjectionCalculator):
            raise TypeError("projection_calculator is not of type ProjectionCalculator")

        #Add rest of options
        self._powder_view.populate_powder_u1(['|Q|'])
        self._powder_view.populate_powder_u2(['Energy'])
        self._main_presenter = None

    def register_master(self, main_view):
        self._main_presenter = main_view.get_presenter()

    def notify(self, command):
        if command == Command.CalculatePowderProjection:
            self._calculate_powder_projection()
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


