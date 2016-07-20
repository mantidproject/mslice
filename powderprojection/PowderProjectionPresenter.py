from command import Command


class PowderProjectionCalculatorPresenter(object):

    def __init__(self,powderView):
        self._powderProjectionView = powderView
        #Add reset of options
        self._powderProjectionView.populate_powder_u1(['Energy'])
        self._powderProjectionView.populate_powder_u2(['|Q|'])

    def notify(self,command):
        if command == Command.CalculatePowderProjection:
            self._powderProjectionView.start_projecting("Projecting _data")


