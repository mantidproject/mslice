from command import Command


class PowderProjectionPresenter(object):

    def __init__(self,mainView,powderView,projection_calculator):
        self._powderProjectionView = powderView
        self._projection_calculator = projection_calculator
        self._workspace_manager_presenter = mainView.get_workspace_manager_presenter()
        #Add reset of options
        self._powderProjectionView.populate_powder_u1(['Energy'])
        self._powderProjectionView.populate_powder_u2(['|Q|'])

    def notify(self,command):
        if command == Command.CalculatePowderProjection:
            axis1 = self._powderProjectionView.get_powder_u1()
            axis2 = self._powderProjectionView.get_powder_u2()
            a

