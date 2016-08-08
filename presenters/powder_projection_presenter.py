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
        self._powder_view.populate_powder_u1(['Energy'])
        self._powder_view.populate_powder_u2(['|Q|'])

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
            pass # TODO ask what to do
        if not selected_workspaces:
            pass
            # TODO notify user? Status bar maybe?

        for workspace in selected_workspaces:
            # TODO if spec decides on suffix naming system then this will be calculated by the presenter!
            output_workspace = self._powder_view.get_output_workspace_name()
            binning = self._projection_calculator.calculate_suggested_binning(workspace)
            # TODO should user be able to suggest own binning?
            self._projection_calculator.calculate_projection(workspace, output_workspace, binning, axis1, axis2)
        self._get_main_presenter().update_displayed_workspaces()

    def _get_main_presenter(self):
        # not storing this variable at construction time gives freedom of creating this before creating the
        # the main presenter
        return self._main_view.get_presenter()