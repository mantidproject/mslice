import os.path
import weakref
import io

from qtpy.QtCore import Qt
from qtpy import QtCore, QtGui, QtWidgets, QtPrintSupport
from mslice.util.qt.qapp import (
    QAppThreadCall,
    create_qapp_if_required,
    force_method_calls_to_qapp_thread,
)
from mslice.models.workspacemanager.file_io import get_save_directory
from mslice.models.workspacemanager.workspace_algorithms import save_workspaces
from mslice.plotting.plot_window.plot_window import PlotWindow
from mslice.plotting.plot_window.slice_plot import SlicePlot
from mslice.plotting.plot_window.cut_plot import CutPlot
import mslice.plotting.pyplot as plt
from mslice.plotting.globalfiguremanager import GlobalFigureManager


def release_active_interactive_cuts_on_slice_plots() -> None:
    for each_figure in GlobalFigureManager.all_figures():
        plot_handler = each_figure.plot_handler
        if isinstance(plot_handler, SlicePlot):
            action_icuts = plot_handler.plot_window.action_interactive_cuts
            if not action_icuts.isChecked():
                continue
            plot_handler.toggle_interactive_cuts(True)
            action_icuts.setChecked(False)


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

        self.window = PlotWindow(manager=weakref.proxy(self))
        self.window.setAttribute(Qt.WA_DeleteOnClose, True)
        self.window.resize(800, 600)

        self.plot_handler = None
        self._picking = None
        self._button_pressed = None

        # Need flags here as matplotlib provides no way to access the grid state
        self._xgrid = False
        self._ygrid = False

        self.window.action_keep.triggered.connect(self.report_as_kept)
        self.window.action_make_current.triggered.connect(self.report_as_current)
        self.window.action_save_image.triggered.connect(self.save_plot)
        self.window.action_copy_image.triggered.connect(self.copy_plot)
        self.window.action_print_plot.triggered.connect(self.print_plot)
        self.window.action_plot_options.triggered.connect(self._plot_options)
        self.window.action_toggle_legends.triggered.connect(self._toggle_legend)
        self.button_pressed_connected(True)
        self.picking_connected(True)
        self.window.show()
        self.window.raise_()

    def show(self):
        self.window.show()
        self.window.activateWindow()
        self.window.raise_()
        self.canvas.draw()

    def window_closing(self):
        if self.plot_handler is not None:
            self.plot_handler.window_closing()
        plt.close(self.number)

    def destroy(self):
        self.window.close()

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

    def enable_make_current(self):
        self._current_figs.enable_make_current()

    def disable_make_current(self):
        self._current_figs.disable_make_current()

    def make_current_disabled(self):
        return self._current_figs.make_current_disabled()

    def flag_as_kept(self):
        self.window.flag_as_kept()

    def flag_as_current(self):
        self.window.flag_as_current()

    def add_slice_plot(self, slice_plotter_presenter, workspace):
        release_active_interactive_cuts_on_slice_plots()
        if self.plot_handler is None:
            # Move the top right corner of all slice plot windows to the left of the screen centre by 1.05
            # the window width and above the screen center by half the window height to prevent cuts/interactive cuts
            # and slices from overlapping
            self.move_window(
                int(self.window.width() * 1.05), int(self.window.height() / 2)
            )
        else:
            self.plot_handler.disconnect(self.window)
        self.plot_handler = SlicePlot(self, slice_plotter_presenter, workspace)
        return self.plot_handler

    def add_cut_plot(self, cut_plotter_presenter, workspace):
        if self.plot_handler is None:
            # Move the top right corner of all cut plot windows to the left of the screen centre by 0.05
            # the window width and above the screen center by half the window height to prevent cuts/interactive cuts
            # and slices from overlapping
            self.move_window(self.window.width() * -0.05, self.window.height() / 2)
        else:
            self.plot_handler.disconnect(self.window)
        self.plot_handler = CutPlot(self, cut_plotter_presenter, workspace)
        return self.plot_handler

    def has_plot_handler(self):
        return self.plot_handler is not None

    def set_cut_background(self, background):
        self.plot_handler.background = background

    def get_cut_background(self):
        return self.plot_handler.background

    def set_is_icut(self, is_icut):
        if self.plot_handler is not None:
            self.plot_handler.set_is_icut(is_icut)

    def picking_connected(self, connect):
        if connect:
            self._picking = self.canvas.mpl_connect("pick_event", self.object_clicked)
        else:
            self.canvas.mpl_disconnect(self._picking)

    def button_pressed_connected(self, connect):
        if connect:
            self._button_pressed = self.canvas.mpl_connect(
                "button_press_event", self.plot_clicked
            )
        else:
            self.canvas.mpl_disconnect(self._button_pressed)

    def _toggle_legend(self):
        axes = self.canvas.figure.gca()

        if not self.plot_handler.show_legends:
            self.plot_handler.show_legends = True
            self.plot_handler.update_legend()

        else:
            self.plot_handler.show_legends = False
            axes.legend_ = None

        self.canvas.draw()

    def plot_clicked(self, event):
        if event.dblclick or event.button == 3:
            self.plot_handler.plot_clicked(event.x, event.y)

    def object_clicked(self, event):
        if event.mouseevent.dblclick or event.mouseevent.button == 3:
            self.plot_handler.object_clicked(event.artist)

    def _plot_options(self):
        self.plot_handler.plot_options()

    def print_plot(self):
        printer = QtPrintSupport.QPrinter()
        printer.setResolution(300)
        printer.setOrientation(QtPrintSupport.QPrinter.Landscape)
        print_dialog = QtPrintSupport.QPrintDialog(printer)
        if print_dialog.exec_():
            pixmap_image = QtWidgets.QWidget.grab(self.canvas)
            page_size = printer.pageRect()
            pixmap_image = pixmap_image.scaled(
                page_size.width(), page_size.height(), Qt.KeepAspectRatio
            )
            painter = QtGui.QPainter(printer)
            painter.drawPixmap(0, 0, pixmap_image)
            painter.end()

    def _get_resolution(self):
        resolution, _ = QtWidgets.QInputDialog.getDouble(
            self.window,
            "Resolution",
            "Enter image resolution (dpi):",
            min=30,
            value=300,
            max=3000,
        )
        return resolution

    def save_plot(self):
        file_path, save_name, ext = get_save_directory(save_as_image=True)
        if file_path is None:
            return
        if hasattr(self.plot_handler, "ws_list"):
            workspaces = self.plot_handler.ws_list
        else:
            if isinstance(self.plot_handler, SlicePlot):
                workspaces = [self.plot_handler.get_cached_workspace()]
            else:
                workspaces = [self.plot_handler.ws_name]
        try:
            save_workspaces(workspaces, file_path, save_name, ext)
        except RuntimeError as e:
            if str(e) == "unrecognised file extension":
                supported_image_types = list(
                    self.window.canvas.get_supported_filetypes().keys()
                )
                if not any([ext.endswith(ft) for ft in supported_image_types]):
                    if ext.endswith("jpg") or ext.endswith("jpeg"):
                        resolution = self._get_resolution()
                        self._save_jpeg_via_qt(resolution, file_path, save_name)
                    else:
                        self.error_box(
                            "Format {} is not supported. "
                            "(Supported formats: {})".format(ext, supported_image_types)
                        )
                elif not save_name.endswith(".pdf"):
                    resolution = self._get_resolution()
                    self.save_image(os.path.join(file_path, save_name), resolution)
                else:
                    self.save_image(os.path.join(file_path, save_name))
            elif str(e) == "dialog cancelled":
                pass
            elif "metadata may be lost" in str(e):
                self.error_box(str(e))
            else:
                raise RuntimeError(e)
        except KeyError:  # Could be case of interactive cuts when the workspace has not been saved yet
            workspace = self.plot_handler.save_icut()
            save_workspaces([workspace], file_path, save_name, ext)

    def save_image(self, path, resolution=300):
        self.canvas.figure.savefig(path, dpi=resolution)

    def _get_figure_image_data(self, resolution=300):
        buf = io.BytesIO()
        self.canvas.figure.savefig(buf, dpi=resolution)
        return buf

    def _save_jpeg_via_qt(self, resolution, file_path, save_name):
        # Use Qt to convert png to jpeg
        buf = self._get_figure_image_data(resolution)
        QtGui.QImage.fromData(buf.getvalue()).save(os.path.join(file_path, save_name))

    def copy_plot(self):
        buf = self._get_figure_image_data()
        QtWidgets.QApplication.clipboard().setImage(
            QtGui.QImage.fromData(buf.getvalue())
        )

    def error_box(self, message):
        error_box = QtWidgets.QMessageBox()
        error_box.setWindowTitle("Error")
        error_box.setIcon(QtWidgets.QMessageBox.Warning)
        error_box.setText(message)
        error_box.exec_()

    def update_grid(self):
        if self._xgrid:
            self.canvas.figure.gca().grid(True, axis="x")
        if self._ygrid:
            self.canvas.figure.gca().grid(True, axis="y")

    def update_axes(self, plot_over, ws_name):
        if self.plot_handler is not None:
            self.plot_handler.on_newplot(plot_over, ws_name)

    def move_window(self, x, y):
        center = QtWidgets.QDesktopWidget().screenGeometry().center()
        self.window.move(int(center.x() - x), int(center.y() - y))

    def get_window_title(self):
        return str(self.window.windowTitle())

    def set_window_title(self, title):
        self.window.setWindowTitle(title)

    @property
    def figure(self):
        return self.canvas.figure

    @property
    def title(self):
        return self.figure.gca().title.get_text()

    @title.setter
    def title(self, value):
        self.figure.gca().title.set_text(value)
        self.window.setWindowTitle(value)

    @property
    def title_size(self):
        return self.figure.gca().title.get_size()

    @title_size.setter
    def title_size(self, value):
        self.figure.gca().title.set_size(value)

    @property
    def x_label(self):
        return self.figure.gca().get_xlabel()

    @x_label.setter
    def x_label(self, value):
        self.figure.gca().set_xlabel(value)

    @property
    def x_label_size(self):
        return self.figure.gca().xaxis.label.get_size()

    @x_label_size.setter
    def x_label_size(self, value):
        self.figure.gca().xaxis.label.set_size(value)

    @property
    def y_label(self):
        return self.figure.gca().get_ylabel()

    @y_label.setter
    def y_label(self, value):
        self.figure.gca().set_ylabel(value)

    @property
    def y_label_size(self):
        return self.figure.gca().yaxis.label.get_size()

    @y_label_size.setter
    def y_label_size(self, value):
        self.figure.gca().yaxis.label.set_size(value)

    @property
    def x_range(self):
        return self.figure.gca().get_xlim()

    @x_range.setter
    def x_range(self, value):
        self.canvas.figure.gca().set_xlim(value)

    @property
    def x_range_font_size(self):
        return self.canvas.figure.gca().xaxis.get_ticklabels()[0].get_size()

    @x_range_font_size.setter
    def x_range_font_size(self, font_size):
        self.canvas.figure.gca().tick_params(axis="x", labelsize=font_size)

    @property
    def y_range(self):
        return self.figure.gca().get_ylim()

    @y_range.setter
    def y_range(self, value):
        self.figure.gca().set_ylim(value)

    @property
    def y_range_font_size(self):
        return self.canvas.figure.gca().yaxis.get_ticklabels()[0].get_size()

    @y_range_font_size.setter
    def y_range_font_size(self, font_size):
        self.canvas.figure.gca().tick_params(axis="y", labelsize=font_size)

    @property
    def x_grid(self):
        return self._xgrid

    @x_grid.setter
    def x_grid(self, value):
        self._xgrid = value
        self.figure.gca().grid(value, axis="x")

    @property
    def y_grid(self):
        return self._ygrid

    @y_grid.setter
    def y_grid(self, value):
        self._ygrid = value
        self.figure.gca().grid(value, axis="y")

    def report_as_current_and_return_previous_status(self):
        last_active_figure_number = None
        disable_make_current_after_plot = False

        if self._current_figs._active_figure is not None:
            last_active_figure_number = self._current_figs.get_active_figure().number
        if self.make_current_disabled():
            self.enable_make_current()
            disable_make_current_after_plot = True
        self.report_as_current()

        return last_active_figure_number, disable_make_current_after_plot

    def reset_current_figure_as_previous(
        self, last_active_figure_number, disable_make_current
    ):
        if last_active_figure_number is not None:
            self._current_figs.set_figure_as_current(last_active_figure_number)
        if disable_make_current:
            self.disable_make_current()


def new_plot_figure_manager(num, global_manager):
    def _new_plot_figure_manager(num, global_manager):
        """Create a new figure manager instance for the given figure.
        Forces all public and non-dunder method calls onto the QApplication thread.
        """
        return force_method_calls_to_qapp_thread(
            PlotFigureManagerQT(num, global_manager)
        )

    create_qapp_if_required()
    # Calls to constructor should be made on the QApplication thread
    return QAppThreadCall(_new_plot_figure_manager)(num, global_manager)
