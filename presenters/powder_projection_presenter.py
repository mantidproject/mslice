from mainview import MainView
from models.projection.powder.projection_calculator import ProjectionCalculator
from views.powder_projection_view import PowderView
from widgets.projection.powder.command import Command


class PowderProjectionPresenter(object):

    def __init__(self,powder_view,main_view,projection_calculator):
        self._powder_view = powder_view
        self._projection_calculator = projection_calculator
        if not isinstance(self._powder_view,PowderView):
            raise TypeError("powder_view is not of type PowderView")
        if not isinstance(self._projection_calculator,ProjectionCalculator):
            raise TypeError("projection_calculator is not of type ProjectionCalculator")
        if not isinstance(main_view,MainView):
            raise TypeError("main_view is not of type MainView")

        self._main_view = main_view
        #Add rest of options
        self._powder_view.populate_powder_u1(['|Q|'])
        self._powder_view.populate_powder_u2(['Energy'])

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
            pass
            raise NotImplementedError('Implement error message')

        for workspace in selected_workspaces:
            self._projection_calculator.calculate_projection(workspace, axis1, axis2)
        self._get_main_presenter().update_displayed_workspaces()

    def _get_main_presenter(self):
        # not storing this variable at construction time gives freedom of creating this before creating the
        # the main presenter
        return self._main_view.get_presenter()