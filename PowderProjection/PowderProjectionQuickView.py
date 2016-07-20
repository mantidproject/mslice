from quickview.QuickView import QuickView
from powderprojection.PowderProjectionCalculatorPresenter import PowderProjectionCalculatorPresenter

class PowderProjectionQuickView(QuickView):
    def __init__(self,commands):
        super(PowderProjectionQuickView,self).__init__(commands)
        self._presenter = PowderProjectionCalculatorPresenter(self)

