import mock
import unittest
from mslice.plotting.plot_window.plot_options import PlotOptionsDialog


class PlotOptionsPresenterTest(unittest.TestCase):
    def setUp(self):
        self.plot_options = mock.create_autospec(PlotOptionsDialog)