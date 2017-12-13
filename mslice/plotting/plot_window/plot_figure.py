from __future__ import (absolute_import, division, print_function)

from matplotlib.backends.backend_qt4 import NavigationToolbar2QT

from PyQt4.QtCore import Qt
from PyQt4 import QtGui
import six

from mslice.plotting.plot_window.slice_plot import SlicePlot
from mslice.plotting.plot_window.cut_plot import CutPlot

from .plot_window_ui import Ui_MainWindow
from .base_qt_plot_window import BaseQtPlotWindow


class PlotFigureManager(BaseQtPlotWindow, Ui_MainWindow):
    def __init__(self, number, manager):
        super(PlotFigureManager, self).__init__(number, manager)

        self.plot_handler = None

        self.actionKeep.triggered.connect(self._report_as_kept_to_manager)
        self.actionMakeCurrent.triggered.connect(self._report_as_current_to_manager)

        self.actionDataCursor.toggled.connect(self.toggle_data_cursor)
        self.stock_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.stock_toolbar.hide()

        self.actionZoom_In.triggered.connect(self.stock_toolbar.zoom)
        self.actionZoom_Out.triggered.connect(self.stock_toolbar.back)
        self.action_save_image.triggered.connect(self.stock_toolbar.save_figure)
        self.action_Print_Plot.triggered.connect(self.print_plot)
        self.actionPlotOptions.triggered.connect(self._plot_options)
        self.actionToggleLegends.triggered.connect(self._toggle_legend)
        self.canvas.mpl_connect('button_press_event', self.plot_clicked)
        self.canvas.mpl_connect('pick_event', self.object_clicked)

        self.show()  # is not a good idea in non interactive mode

    def add_slice_plot(self, slice_plotter):
        self.plot_handler = SlicePlot(self, self.canvas, slice_plotter)

    def add_cut_plot(self, cut_plotter):
        self.plot_handler = CutPlot(self, self.canvas, cut_plotter)

    def _toggle_legend(self):
        self.plot_handler.toggle_legend()

    def plot_clicked(self, event):
        if event.dblclick:
            self.plot_handler.plot_clicked(event.x, event.y)

    def object_clicked(self, event):
        if event.mouseevent.dblclick:
            self.plot_handler.object_clicked(event.artist)

    def toggle_data_cursor(self):
        if self.actionDataCursor.isChecked():
            self.stock_toolbar.message.connect(self.statusbar.showMessage)
            self.canvas.setCursor(Qt.CrossCursor)

        else:
            self.stock_toolbar.message.disconnect()
            self.canvas.setCursor(Qt.ArrowCursor)

    def _display_status(self, status):
        if status == "kept":
            self.actionKeep.setChecked(True)
            self.actionMakeCurrent.setChecked(False)
        elif status == "current":
            self.actionMakeCurrent.setChecked(True)
            self.actionKeep.setChecked(False)

    def _plot_options(self):
        self.plot_handler.plot_options()

    def print_plot(self):
        printer = QtGui.QPrinter()
        printer.setResolution(300)
        printer.setOrientation(QtGui.QPrinter.Landscape) #  landscape by default
        print_dialog = QtGui.QPrintDialog(printer)
        if print_dialog.exec_():
            pixmap_image = QtGui.QPixmap.grabWidget(self.canvas)
            page_size = printer.pageRect()
            pixmap_image = pixmap_image.scaled(page_size.width(), page_size.height(), Qt.KeepAspectRatio)
            painter = QtGui.QPainter(printer)
            painter.drawPixmap(0,0,pixmap_image)
            painter.end()

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
