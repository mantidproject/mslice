import unittest
import mock
from mock import call
from powderprojection.PowderProjectionPresenter import PowderProjectionPresenter
from powderprojection.PowderProjectionView import PowderView
from powderprojection.ProjectionCalculator import ProjectionCalculator
from mainview import MainView

class PowderProjectionPresenterTest(unittest.TestCase):
    def setUp(self):
        self.view = mock.create_autospec(PowderView)
        self.projection_calculator = mock.create_autospec(ProjectionCalculator)
        self.mainview = mock.create_autospec(MainView)

    def test_constructor_success(self):
        self.presenter = PowderProjectionPresenter(self.view,self.mainview,self.projection_calculator)