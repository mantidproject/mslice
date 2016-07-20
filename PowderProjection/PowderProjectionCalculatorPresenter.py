from command import Command


class PowderProjectionCalculatorPresenter(object):

    def __init__(self,powderView):
        self._powderProjectionView = powderView

    def notify(self,command):
        if command == Command.CalculatePowderProjection:
            self._powderProjectionView.start_projecting("Projecting data")


