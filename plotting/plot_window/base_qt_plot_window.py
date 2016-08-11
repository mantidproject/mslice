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
        self._new_axes_limits = [[None,None],[None,None]] # For zooming [[x_lower,x_upper],[y_lower,y_upper]]
        # TODO implement zoom history stack
        self._zoom_history_stack = []
        self._zoom_rect = None

    def zoom_out(self):
        if self._zoom_history_stack:
            old_dim = self._zoom_history_stack.pop()
            self.axes.set_xlim(*old_dim[0])
            self.axes.set_ylim(*old_dim[1])
            self.draw()

    def zoom_in(self):
        self._zoom_history_stack.append([self.axes.get_xlim(), self.axes.get_ylim()])
        self._rect_cid = self.mpl_connect("button_press_event", self._start_draw_zoom_rect)
        self.setCursor(QtCore.Qt.CrossCursor)

    def _start_draw_zoom_rect(self,mouse_event):
        self.mpl_disconnect(self._rect_cid)
        self._motion_cid = self.mpl_connect("motion_notify_event",self._redraw_zoom_rect)
        self._end_zoom_cid = self.mpl_connect("button_release_event", self._end_draw_zoom_rect)
        self._new_axes_limits[0][0] = mouse_event.xdata
        self._new_axes_limits[1][0] = mouse_event.ydata


    def _redraw_zoom_rect(self,mouse_event):
        self._new_axes_limits[0][1] = mouse_event.xdata
        self._new_axes_limits[1][1] = mouse_event.ydata
        if self._zoom_rect:
            self._zoom_rect.remove()
        x,y = self._new_axes_limits[0][0], self._new_axes_limits[1][0]
        width, height = map(lambda x: x[1]-x[0], self._new_axes_limits)
        self._zoom_rect = Rectangle((x, y), width, height, fill=False)
        self.axes.add_patch(self._zoom_rect)
        self.draw()


    def _end_draw_zoom_rect(self,mouse_event):
        self.mpl_disconnect(self._motion_cid)
        self.mpl_disconnect(self._end_zoom_cid)
        self._new_axes_limits[0].sort()
        self._new_axes_limits[1].sort()
        self.axes.set_xlim(*self._new_axes_limits[0])
        self.axes.set_ylim(*self._new_axes_limits[1])
        self._new_axes_limits = [[None,None],[None,None]]
        self.setCursor(QtCore.Qt.ArrowCursor)
        if self._zoom_rect:
            self._zoom_rect.remove()
            self._zoom_rect = None
        self.draw()


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