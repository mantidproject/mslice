from command import Command


class PowderProjectionPresenter(object):

    def __init__(self,powderView,mainView,projection_calculator):
        self._powderProjectionView = powderView
        self._projection_calculator = projection_calculator
        self._workspace_manager_presenter = mainView.get_workspace_manager_presenter()
        #Add reset of options
        self._powderProjectionView.populate_powder_u1(['Energy'])
        self._powderProjectionView.populate_powder_u2(['|Q|'])

    def notify(self,command):
        if command == Command.CalculatePowderProjection:
            selected_workspaces = self._workspace_manager_presenter.get_selected_workspaces()
            axis1 = self._powderProjectionView.get_powder_u1()
            axis2 = self._powderProjectionView.get_powder_u2()
            if not selected_workspaces:
                pass
                #TODO notify user? Status bar maybe?

            for workspace in selected_workspaces:
                output_workspace = self._powderProjectionView.get_output_workspace_name()
                binning = self._projection_calculator.CalculateSuggestedBinning(workspace)
                binning = '1,1,10' #TODO REMOVE THIS
                #TODO should user be able to suggest own binning?
                self._projection_calculator.CalculateProjections(workspace, output_workspace, binning, axis1, axis2)
            self._workspace_manager_presenter.refresh()
