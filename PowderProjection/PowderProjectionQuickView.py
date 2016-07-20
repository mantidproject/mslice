from QuickView.QuickView import QuickView
from PowderProjection.PowderProjectionCalculatorPresenter import PowderProjectionCalculatorPresenter

class PowderProjectionQuickView(QuickView):
    def __init__(self,commands):
        super(PowderProjectionQuickView,self).__init__(commands)
        self._presenter = PowderProjectionCalculatorPresenter(self)

