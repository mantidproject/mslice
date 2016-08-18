from PyQt4 import QtGui, QtCore
from base_plot_window import BasePlotWindow
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

class MatplotlibCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        self.axes.hold(False)
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self._rect_cid = None
        self._motion_cid = None
        self._new_axes_limits = [[None, None], [None, None]] # For zooming [[x_lower,x_upper],[y_lower,y_upper]]
        self._zoom_history_stack = []




class BaseQtPlotWindow(BasePlotWindow, QtGui.QMainWindow):
    """Inherit from this and a Ui_MainWindow from QT Designer to get a working PlotWindow

    The central widget will be replaced by the canvas"""
    def __init__(self, number, manager):
        super(BaseQtPlotWindow,self).__init__(number,manager)
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        self.canvas = MatplotlibCanvas(self)
        self.setCentralWidget(self.canvas)
        self.setWindowTitle('Figure %i'%number)

        self.show() # this isn,t a good idea in non interactive mode #TODO FIX IT

    def closeEvent(self, event):
        self._manager.figure_closed(self.number)
        event.accept()