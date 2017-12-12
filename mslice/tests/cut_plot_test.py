from mock import MagicMock, PropertyMock
import unittest

from mslice.plotting.plot_window.cut_plot import CutPlot


class CutPlotTest(unittest.TestCase):

    def setUp(self):
        self.plot_figure = MagicMock()
