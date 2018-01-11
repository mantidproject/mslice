from __future__ import (absolute_import, division, print_function)

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy import QtWidgets

from mslice.util.qt import load_ui
from .base_plot_window import BasePlotWindow


class MatplotlibCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class BaseQtPlotWindow(BasePlotWindow, QtWidgets.QMainWindow):
    """Inherit from this and a Ui_MainWindow from QT Designer to get a working PlotWindow

    The central widget will be replaced by the canvas"""
    def __init__(self, number, manager):
        QtWidgets.QMainWindow.__init__(self, None)
        BasePlotWindow.__init__(self, number, manager)
        load_ui(__file__, 'plot_window.ui', self)
        self.canvas = MatplotlibCanvas(self)
        self.canvas.manager = self
        self.setCentralWidget(self.canvas)

    def closeEvent(self, event):
        self._manager.figure_closed(self.number)
        event.accept()
