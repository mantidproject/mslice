from quickview.QuickView import QuickView
from powderprojection.PowderProjectionPresenter import PowderProjectionPresenter

class PowderProjectionQuickView(QuickView):
    def __init__(self,commands):
        super(PowderProjectionQuickView,self).__init__(commands)
        self._presenter = PowderProjectionPresenter(self)

