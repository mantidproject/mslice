
import os.path
import weakref

import six
from mslice.util.qt.QtCore import Qt
from mslice.util.qt import QtCore, QtGui, QtWidgets

from mslice.models.workspacemanager.file_io import get_save_directory
from mslice.models.workspacemanager.workspace_algorithms import save_workspaces
from mslice.plotting.plot_window.plot_window import PlotWindow
from mslice.plotting.plot_window.slice_plot import SlicePlot
from mslice.plotting.plot_window.cut_plot import CutPlot
import mslice.plotting.pyplot as plt


class PlotFigureManagerQT(QtCore.QObject):
    """Manage a Qt window along with the keep/make current status"""

    def __init__(self, number, current_figs):
        """
        Initialize the object with a figure number and its manager
        :param number: The number of the figure
        :param current_figs: A reference to the global manager for all figures
        """
        super(PlotFigureManagerQT, self).__init__()
        self.number = number
        self._current_figs = current_figs

        # window instance
        self.window = PlotWindow(manager=weakref.proxy(self))
        self.window.resize(800, 600)

        self._plot_handler = None
        self._picking = None

        # Need flags here as matplotlib provides no way to access the grid state
        self._xgrid = False
        self._ygrid = False

        self.window.action_keep.triggered.connect(self.report_as_kept)
        self.window.action_make_current.triggered.connect(self.report_as_current)
        self.window.action_save_image.triggered.connect(self.save_plot)
        self.window.action_print_plot.triggered.connect(self.print_plot)
        self.window.action_plot_options.triggered.connect(self._plot_options)
        self.window.action_toggle_legends.triggered.connect(self._toggle_legend)
        self.canvas.mpl_connect('button_press_event', self.plot_clicked)
        self.picking_connected(True)

        self.show()

    def show(self):
        self.window.show()
        self.window.activateWindow()
        self.window.raise_()

    def window_closing(self):
        if self._plot_handler is not None:
            self._plot_handler.window_closing()
        plt.close(self.number)

    def resize(self, width, height):
        self.window.resize(width, height)

    @property
    def canvas(self):
        return self.window.canvas

    def report_as_kept(self):
        """Report to the global figure manager that this figure should be kept"""
        self._current_figs.set_figure_as_kept(self.number)

    def report_as_current(self):
        """Report to the global figure manager that this figure should be made active"""
        self._current_figs.set_figure_as_current(self.number)

    def flag_as_kept(self):
        self.window.flag_as_kept()

    def flag_as_current(self):
        self.window.flag_as_current()

    def add_slice_plot(self, slice_plotter, workspace):
        if self._plot_handler is None:
            self.move_window(-self.window.width() / 2, 0)
        else:
            self._plot_handler.disconnect(self.window)
        self._plot_handler = SlicePlot(self, slice_plotter, workspace)

    def add_cut_plot(self, cut_plotter_presenter, workspace):
        if self._plot_handler is None:
            self.move_window(self.window.width() / 2, 0)
        else:
            self._plot_handler.disconnect(self.window)
        self._plot_handler = CutPlot(self, cut_plotter_presenter, workspace)

    def has_plot_handler(self):
        return self._plot_handler is not None

    def set_cut_background(self, background):
        self._plot_handler.background = background

    def get_cut_background(self):
        return self._plot_handler.background

    def is_icut(self, is_icut):
        if self._plot_handler is not None:
            self._plot_handler.is_icut(is_icut)

    def picking_connected(self, connect):
        if connect:
            self._picking = self.canvas.mpl_connect('pick_event', self.object_clicked)
        else:
            self.canvas.mpl_disconnect(self._picking)

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

    def _plot_options(self):
        self._plot_handler.plot_options()

    def print_plot(self):
        printer = QtWidgets.QPrinter()
        printer.setResolution(300)
        printer.setOrientation(QtWidgets.QPrinter.Landscape)
        print_dialog = QtWidgets.QPrintDialog(printer)
        if print_dialog.exec_():
            pixmap_image = QtGui.QPixmap.grabWidget(self.canvas)
            page_size = printer.pageRect()
            pixmap_image = pixmap_image.scaled(page_size.width(), page_size.height(), Qt.KeepAspectRatio)
            painter = QtGui.QPainter(printer)
            painter.drawPixmap(0,0,pixmap_image)
            painter.end()

    def save_plot(self):
        file_path, save_name, ext = get_save_directory(save_as_image=True)
        workspace = self._plot_handler.ws_name
        try:
            save_workspaces([workspace], file_path, save_name, ext, slice_nonpsd=True)
        except RuntimeError as e:
            if str(e) == "unrecognised file extension":
                self.save_image(os.path.join(file_path, save_name))
            elif str(e) == "dialog cancelled":
                pass
            else:
                raise RuntimeError(e)
        except KeyError:   # Could be case of interactive cuts when the workspace has not been saved yet
            workspace = self._plot_handler.save_icut()
            save_workspaces([workspace], file_path, save_name, ext)

    def save_image(self, path):
        self.canvas.figure.savefig(path)

    def error_box(self, message):
        error_box = QtWidgets.QMessageBox(self)
        error_box.setWindowTitle("Error")
        error_box.setIcon(QtWidgets.QMessageBox.Warning)
        error_box.setText(message)
        error_box.show()

    def update_grid(self):
        if self._xgrid:
            self.canvas.figure.gca().grid(True, axis='x')
        if self._ygrid:
            self.canvas.figure.gca().grid(True, axis='y')

    def move_window(self, x, y):
        window = self.window
        window.move(window.pos().x() + x, window.pos().y() + y)

    def get_window_title(self):
        return six.text_type(self.window.windowTitle())

    def set_window_title(self, title):
        self.window.setWindowTitle(title)

    @property
    def figure(self):
        return self.canvas.figure

    @property
    def title(self):
        return self.figure.gca().get_title()

    @title.setter
    def title(self, value):
        self.figure.gca().set_title(value)
        self.window.setWindowTitle(value)

    @property
    def x_label(self):
        return self.figure.gca().get_xlabel()

    @x_label.setter
    def x_label(self, value):
        self.figure.gca().set_xlabel(value)

    @property
    def y_label(self):
        return self.figure.gca().get_ylabel()

    @y_label.setter
    def y_label(self, value):
        self.figure.gca().set_ylabel(value)

    @property
    def x_range(self):
        return self.figure.gca().get_xlim()

    @x_range.setter
    def x_range(self, value):
        self.canvas.figure.gca().set_xlim(value)

    @property
    def y_range(self):
        return self.figure.gca().get_ylim()

    @y_range.setter
    def y_range(self, value):
        self.figure.gca().set_ylim(value)

    @property
    def x_grid(self):
        return self._xgrid

    @x_grid.setter
    def x_grid(self, value):
        self._xgrid = value
        self.figure.gca().grid(value, axis='x')

    @property
    def y_grid(self):
        return self._ygrid

    @y_grid.setter
    def y_grid(self, value):
        self._ygrid = value
        self.figure.gca().grid(value, axis='y')
