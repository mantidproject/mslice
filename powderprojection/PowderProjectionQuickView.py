from quickview.QuickView import QuickView
from powderprojection.PowderProjectionPresenter import PowderProjectionPresenter
from powderprojection.MantidProjectionCalculator import MantidProjectionCalculator
from powderprojection.PowderProjectionView import PowderView


class PowderProjectionQuickView(QuickView,PowderView):
    def __init__(self,main_view,commands):
        super(PowderProjectionQuickView,self).__init__(commands)
        proj_calculator = MantidProjectionCalculator()
        self._presenter = PowderProjectionPresenter(self, main_view,proj_calculator)

    def __getattribute__(self, item):
        # This is needed to handle calls to GUI functions generated on the fly correctly
        object.__getattr__(self, item)