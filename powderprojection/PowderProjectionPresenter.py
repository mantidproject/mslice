from command import Command
from powderprojection.PowderProjectionView import PowderView
from powderprojection.ProjectionCalculator import  ProjectionCalculator
from mainview import MainView


class PowderProjectionPresenter(object):

    def __init__(self,powder_view,main_view,projection_calculator):
        self._powder_view = powder_view
        self._projection_calculator = projection_calculator
        if not isinstance(self._powder_view,PowderView):
            raise ValueError("powder_view is not of type PowderView")
        if not isinstance(self._projection_calculator,ProjectionCalculator):
            raise ValueError("projection_calculator is not of type ProjectionCalculator")
        if not isinstance(main_view,MainView):
            raise ValueError("main_view is not of type MainView")

        self._main_presenter = main_view.get_presenter()
        #Add rest of options
        self._powder_view.populate_powder_u1(['Energy'])
        self._powder_view.populate_powder_u2(['|Q|'])

    def notify(self,command):
        if command == Command.CalculatePowderProjection:
            selected_workspaces = self._main_presenter.get_selected_workspaces()
            axis1 = self._powder_view.get_powder_u1()
            axis2 = self._powder_view.get_powder_u2()
            if not selected_workspaces:
                pass
                #TODO notify user? Status bar maybe?

            for workspace in selected_workspaces:
                output_workspace = self._powder_view.get_output_workspace_name()
                binning = self._projection_calculator.calculate_suggested_binning(workspace)
                #TODO should user be able to suggest own binning?
                self._projection_calculator.calculate_projections(workspace, output_workspace, binning, axis1, axis2)
            self._main_presenter.update_displayed_workspaces()
