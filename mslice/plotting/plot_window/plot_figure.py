from __future__ import (absolute_import, division, print_function)

import importlib

from matplotlib.figure import Figure
import six
from mslice.util.qt import QT_VERSION
from mslice.util.qt.QtCore import Qt
from mslice.util.qt import QtWidgets

from mslice.plotting.plot_window.slice_plot import SlicePlot
from mslice.plotting.plot_window.cut_plot import CutPlot
from mslice.util.qt import load_ui
from mslice.plotting.plot_window.base_plot_window import BasePlotWindow

# The FigureCanvas & Toolbar are QWidgets so we must import it from the mpl backend that matches
# the version of Qt we are running with
mpl_qt_backend  = importlib.import_module('matplotlib.backends.backend_qt{}agg'.format(QT_VERSION[0]))
FigureCanvasQTAgg = getattr(mpl_qt_backend, 'FigureCanvasQTAgg')
NavigationToolbar2QT = getattr(mpl_qt_backend, 'NavigationToolbar2QT')


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvasQTAgg.__init__(self, self.figure)
        self.setParent(parent)

        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtWidgets.QSizePolicy.Expanding,
                                        QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)


PlotWindowUI, _ = load_ui(__file__, 'plot_window.ui')


class PlotFigureManager(BasePlotWindow, PlotWindowUI, QtWidgets.QMainWindow):
    def __init__(self, number, manager):
        QtWidgets.QMainWindow.__init__(self)
        BasePlotWindow.__init__(self, number, manager)
        self.setupUi(self)

        self.canvas = MatplotlibCanvas(self)
        self.canvas.manager = self
        self.setCentralWidget(self.canvas)

        self._plot_handler = None
        self.picking = None

        # Need flags here as matplotlib provides no way to access the grid state
        self._xgrid = False
        self._ygrid = False

        self.actionKeep.triggered.connect(self._report_as_kept_to_manager)
        self.actionMakeCurrent.triggered.connect(self._report_as_current_to_manager)

        self.stock_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.stock_toolbar.message.connect(self.statusbar.showMessage)
        self.stock_toolbar.hide()

        self.actionZoom_In.triggered.connect(self.stock_toolbar.zoom)
        self.actionZoom_Out.triggered.connect(self.stock_toolbar.back)
        self.action_save_image.triggered.connect(self.stock_toolbar.save_figure)
        self.action_Print_Plot.triggered.connect(self.print_plot)
        self.actionPlotOptions.triggered.connect(self._plot_options)
        self.actionToggleLegends.triggered.connect(self._toggle_legend)
        self.actionInteractive_Cuts.setVisible(False)
        self.actionSave_Cut.setVisible(False)
        self.canvas.mpl_connect('button_press_event', self.plot_clicked)
        self.picking_connected(True)

        self.show()  # is not a good idea in non interactive mode

    def add_slice_plot(self, slice_plotter):
        self._plot_handler = SlicePlot(self, self.canvas, slice_plotter)

    def add_cut_plot(self, cut_plotter):
        self._plot_handler = CutPlot(self, self.canvas, cut_plotter)

    def set_as_icut(self):
        self._plot_handler.set_as_icut()

    def picking_connected(self, connect):
        if connect:
            self.picking = self.canvas.mpl_connect('pick_event', self.object_clicked)
        else:
            self.canvas.mpl_disconnect(self.picking)

    def _toggle_legend(self):
        axes = self.canvas.figure.gca()
        if axes.legend_ is None:
            self._plot_handler.update_legend()
        else:
            axes.legend_ = None
        self.canvas.draw()

    def plot_clicked(self, event):
        if event.dblclick or event.button == 3:
            self._plot_handler.plot_clicked(event.x, event.y)

    def object_clicked(self, event):
        if event.mouseevent.dblclick or event.mouseevent.button == 3:
            self._plot_handler.object_clicked(event.artist)

    def _display_status(self, status):
        if status == "kept":
            self.actionKeep.setChecked(True)
            self.actionMakeCurrent.setChecked(False)
        elif status == "current":
            self.actionMakeCurrent.setChecked(True)
            self.actionKeep.setChecked(False)

    def _plot_options(self):
        self._plot_handler.plot_options()

    def print_plot(self):
        printer = QtWidgets.QPrinter()
        printer.setResolution(300)
        printer.setOrientation(QtWidgets.QPrinter.Landscape) #  landscape by default
        print_dialog = QtWidgets.QPrintDialog(printer)
        if print_dialog.exec_():
            pixmap_image = QtWidgets.QPixmap.grabWidget(self.canvas)
            page_size = printer.pageRect()
            pixmap_image = pixmap_image.scaled(page_size.width(), page_size.height(), Qt.KeepAspectRatio)
            painter = QtWidgets.QPainter(printer)
            painter.drawPixmap(0,0,pixmap_image)
            painter.end()

    def update_grid(self):
        if self._xgrid:
            self.canvas.figure.gca().grid(True, axis='x')
        if self._ygrid:
            self.canvas.figure.gca().grid(True, axis='y')

    def get_window_title(self):
        return six.text_type(self.windowTitle())

    def set_window_title(self, title):
        self.setWindowTitle(title)

    @property
    def title(self):
        return self.canvas.figure.gca().get_title()

    @title.setter
    def title(self, value):
        self.canvas.figure.gca().set_title(value)
        self.setWindowTitle(value)

    @property
    def x_label(self):
        return self.canvas.figure.gca().get_xlabel()

    @x_label.setter
    def x_label(self, value):
        self.canvas.figure.gca().set_xlabel(value)

    @property
    def y_label(self):
        return self.canvas.figure.gca().get_ylabel()

    @y_label.setter
    def y_label(self, value):
        self.canvas.figure.gca().set_ylabel(value)

    @property
    def x_range(self):
        return self.canvas.figure.gca().get_xlim()

    @x_range.setter
    def x_range(self, value):
        self.canvas.figure.gca().set_xlim(value)

    @property
    def y_range(self):
        return self.canvas.figure.gca().get_ylim()

    @y_range.setter
    def y_range(self, value):
        self.canvas.figure.gca().set_ylim(value)

    @property
    def x_grid(self):
        return self._xgrid

    @x_grid.setter
    def x_grid(self, value):
        self._xgrid = value
        self.canvas.figure.gca().grid(value, axis='x')

    @property
    def y_grid(self):
        return self._ygrid

    @y_grid.setter
    def y_grid(self, value):
        self._ygrid = value
        self.canvas.figure.gca().grid(value, axis='y')
