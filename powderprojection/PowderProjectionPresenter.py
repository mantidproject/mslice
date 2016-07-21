from command import Command


class PowderProjectionPresenter(object):

    def __init__(self,powder_view,main_view,projection_calculator):
        self.powder_prejection_calculator = powder_view
        self._projection_calculator = projection_calculator
        self._main_presenter = main_view.get_presenter()
        #Add rest of options
        self.powder_prejection_calculator.populate_powder_u1(['Energy'])
        self.powder_prejection_calculator.populate_powder_u2(['|Q|'])

    def notify(self,command):
        if command == Command.CalculatePowderProjection:
            selected_workspaces = self._main_presenter.get_selected_workspaces()
            axis1 = self.powder_prejection_calculator.get_powder_u1()
            axis2 = self.powder_prejection_calculator.get_powder_u2()
            if not selected_workspaces:
                pass
                #TODO notify user? Status bar maybe?

            for workspace in selected_workspaces:
                output_workspace = self.powder_prejection_calculator.get_output_workspace_name()
                binning = self._projection_calculator.CalculateSuggestedBinning(workspace)
                #TODO should user be able to suggest own binning?
                self._projection_calculator.CalculateProjections(workspace, output_workspace, binning, axis1, axis2)
            self._main_presenter.refresh()
