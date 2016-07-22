from quickview.QuickView import QuickView
from powderprojection.PowderProjectionPresenter import PowderProjectionPresenter
from powderprojection.MantidProjectionCalculator import MantidProjectionCalculator
from powderprojection.PowderProjectionView import PowderView


class PowderProjectionQuickView(QuickView,PowderView):
    def __init__(self,mainView,commands):
        super(PowderProjectionQuickView,self).__init__(commands)
        proj_calculator = MantidProjectionCalculator()
        self._presenter = PowderProjectionPresenter(self,mainView,proj_calculator)
