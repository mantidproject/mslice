import mock
from mock import MagicMock, PropertyMock
import unittest
from PyQt4.QtCore import pyqtSignal
from PyQt4.QtTest import QTest
from mslice.plotting.plot_window.plot_options import CutPlotOptions, SlicePlotOptions
from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter, SlicePlotOptionsPresenter
from mslice.plotting.plot_window.plot_figure import PlotFigureManager



class CutPlotOptionsPresenterTest(unittest.TestCase):
    def setUp(self):
        self.view = MagicMock()
        self.model = MagicMock()

        self.view_title = ""

    def test_changeTitle(self):

        def get_model_title():
            return self.model_title

        def set_model_title(value):
            self.model_title = value

        def get_view_title(ob, instance, owner):
            return self.view_title

        def set_view_title(ob, instance, value):
            self.view_title = value

        self.model.canvas.figure.gca().get_title = get_model_title
        self.model.canvas.figure.gca().set_title = set_model_title

        title_property = PropertyMock()
        title_property.__get__ = get_view_title
        title_property.__set__ = set_view_title
        type(self.view).title = title_property

        #test title passed model -> view
        self.model_title = "Title 0"
        self.presenter = CutPlotOptionsPresenter(self.view, self.model)
        self.assertEquals(self.view_title, "Title 0")

        #test title passed view -> model
        self.view.title = "title 1"
        self.presenter._value_modified('title')
        self.presenter.get_new_config()
        #self.assertEquals(self.presenter.title, "title 1")









