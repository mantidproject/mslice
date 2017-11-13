from __future__ import (absolute_import, division, print_function)
from functools import partial
from itertools import chain

from mantid.simpleapi import AnalysisDataService
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from matplotlib.container import ErrorbarContainer
import matplotlib.colors as colors
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QInputDialog, QPrinter, QPrintDialog, QPixmap, QPainter
import numpy as np
import six

from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter, SlicePlotOptionsPresenter

from .plot_window_ui import Ui_MainWindow
from .base_qt_plot_window import BaseQtPlotWindow
from .plot_options import SlicePlotOptions, CutPlotOptions


class PlotFigureManager(BaseQtPlotWindow, Ui_MainWindow):
    def __init__(self, number, manager):
        super(PlotFigureManager, self).__init__(number, manager)

        self.legends_shown = True
        self.legends_visible = []
        self.lines_visible = {}
        self.slice_plotter = None
        self.workspace_title = None
        self.menuIntensity.setDisabled(True)

        self.actionKeep.triggered.connect(self._report_as_kept_to_manager)
        self.actionMakeCurrent.triggered.connect(self._report_as_current_to_manager)
        self.actionDump_To_Console.triggered.connect(self._dump_script_to_console)

        self.actionDataCursor.toggled.connect(self.toggle_data_cursor)
        self.stock_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.stock_toolbar.hide()

        self.actionZoom_In.triggered.connect(self.stock_toolbar.zoom)
        self.actionZoom_Out.triggered.connect(self.stock_toolbar.back)
        self.action_save_image.triggered.connect(self.stock_toolbar.save_figure)
        self.action_Print_Plot.triggered.connect(self.print_plot)
        self.actionPlotOptions.triggered.connect(self._plot_options)
        self.actionToggleLegends.triggered.connect(self._toggle_legend)

        self.show()  # is not a good idea in non interactive mode

    def is_slice_figure(self):
        return bool(self.canvas.figure.gca().get_images())

    def is_cut_figure(self):
        return not bool(self.canvas.figure.gca().get_images())

    def add_slice_plotter(self, slice_plotter):
        self.slice_plotter = slice_plotter
        self.menuIntensity.setDisabled(False)
        self.ws_title = self.title
        self.actionS_Q_E.triggered.connect(partial(self.show_intensity_plot, self.actionS_Q_E,
                                                   self.slice_plotter.show_scattering_function, False))
        self.actionChi_Q_E.triggered.connect(partial(self.show_intensity_plot, self.actionChi_Q_E,
                                                     self.slice_plotter.show_dynamical_susceptibility, True))
        self.actionChi_Q_E_magnetic.triggered.connect(partial(self.show_intensity_plot, self.actionChi_Q_E_magnetic,
                                                              self.slice_plotter.show_dynamical_susceptibility_magnetic,
                                                              True))

    def intensity_selection(self, selected):
        '''Ticks selected and un-ticks other intensity options. Returns previous selection'''
        options = self.menuIntensity.actions()
        previous = None
        for op in options:
            if op.isChecked() and op is not selected:
                previous = op
            op.setChecked(False)
        selected.setChecked(True)
        return previous

    def show_intensity_plot(self, action, slice_plotter_method, temp_dependent):
        if action.isChecked():
            previous = self.intensity_selection(action)
            cbar_log = self.colorbar_log
            x_range = self.x_range
            y_range = self.y_range
            title = self.title
            if temp_dependent:
                if not self._run_temp_dependent(slice_plotter_method, previous):
                    return
            else:
                slice_plotter_method(self.ws_title)
            self.change_slice_plot(self.colorbar_range, cbar_log)
            self.x_range = x_range
            self.y_range = y_range
            self.title = title
            self.canvas.draw()
        else:
            action.setChecked(True)

    def _run_temp_dependent(self, slice_plotter_method, previous):
        try:
            slice_plotter_method(self.ws_title)
        except ValueError:  # sample temperature not yet set
            try:
                field = self.ask_sample_temperature_field(str(self.ws_title))
            except RuntimeError:  # if cancel is clicked, go back to previous selection
                self.intensity_selection(previous)
                return False
            self.slice_plotter.add_sample_temperature_field(field)
            self.slice_plotter.update_sample_temperature(self.ws_title)
            slice_plotter_method(self.ws_title)
        return True

    def ask_sample_temperature_field(self, ws_name):
        if ws_name[-3:] == '_QE':
            ws_name = ws_name[:-3]
        ws = AnalysisDataService[ws_name]
        temp_field, confirm = QInputDialog.getItem(self, 'Sample Temperature',
                                                   'Sample Temperature not found. Select the sample temperature field:',
                                                   ws.run().keys(), False)
        if not confirm:
            raise RuntimeError("sample_temperature_dialog cancelled")
        else:
            return str(temp_field)

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
        if self.is_slice_figure():
            view_class, presenter_class = SlicePlotOptions, SlicePlotOptionsPresenter
        else:
            view_class, presenter_class = CutPlotOptions, CutPlotOptionsPresenter
        new_config = presenter_class(view_class(), self).get_new_config()
        if new_config:
            self.canvas.draw()

    def print_plot(self):
        printer = QPrinter()
        printer.setResolution(300)
        printer.setOrientation(QPrinter.Landscape) #  landscape by default
        print_dialog = QPrintDialog(printer)
        if print_dialog.exec_():
            pixmap_image = QPixmap.grabWidget(self.canvas)
            page_size = printer.pageRect()
            pixmap_image = pixmap_image.scaled(page_size.width(), page_size.height(), Qt.KeepAspectRatio)
            painter = QPainter(printer)
            painter.drawPixmap(0,0,pixmap_image)
            painter.end()

    @staticmethod
    def get_min(data, absolute_minimum=-np.inf):
        """Determines the minimum of a set of numpy arrays"""
        data = data if isinstance(data, list) else [data]
        running_min = []
        for values in data:
            try:
                running_min.append(np.min(values[np.isfinite(values) * (values > absolute_minimum)]))
            except ValueError:  # If data is empty or not array of numbers
                pass
        return np.min(running_min) if running_min else absolute_minimum

    def change_cut_plot(self, xy_config):
        current_axis = self.canvas.figure.gca()
        if xy_config['x_log']:
            xdata = [ll.get_xdata() for ll in current_axis.get_lines()]
            xmin = self.get_min(xdata, absolute_minimum=0.)
            current_axis.set_xscale('symlog', linthreshx=pow(10, np.floor(np.log10(xmin))))
            if xmin > 0:
                xy_config['x_range'] = (xmin, xy_config['x_range'][1])
        else:
            current_axis.set_xscale('linear')
        if xy_config['y_log']:
            ydata = [ll.get_ydata() for ll in current_axis.get_lines()]
            ymin = self.get_min(ydata, absolute_minimum=0.)
            current_axis.set_yscale('symlog', linthreshy=pow(10, np.floor(np.log10(ymin))))
            if ymin > 0:
                xy_config['y_range'] = (ymin, xy_config['y_range'][1])
        else:
            current_axis.set_yscale('linear')
        self.x_range = xy_config['x_range']
        self.y_range = xy_config['y_range']

    def change_slice_plot(self, colorbar_range, logarithmic):
        current_axis = self.canvas.figure.gca()
        images = current_axis.get_images()
        if len(images) != 1:
            raise RuntimeError("Expected single image on axes, found " + str(len(images)))
        mappable = images[0]
        vmin, vmax = colorbar_range
        if logarithmic and type(mappable.norm) != colors.LogNorm:
            mappable.colorbar.remove()
            if vmin == float(0):
                vmin = 0.001
            norm = colors.LogNorm(vmin, vmax)
            mappable.set_norm(norm)
            self.canvas.figure.colorbar(mappable)
        elif not logarithmic and type(mappable.norm) != colors.Normalize:
            mappable.colorbar.remove()
            norm = colors.Normalize(vmin, vmax)
            mappable.set_norm(norm)
            self.canvas.figure.colorbar(mappable)
        mappable.set_clim((vmin, vmax))

    def _has_errorbars(self):
        """True current axes has visible errorbars,
         False if current axes has hidden errorbars"""
        current_axis = self.canvas.figure.gca()
        # If all the error bars have alpha= 0 they are all transparent (hidden)
        containers = [x for x in current_axis.containers if isinstance(x, ErrorbarContainer)]
        line_components = [x.get_children() for x in containers]
        # drop the first element of each container because it is the the actual line
        errorbars = [x[1:] for x in line_components]
        errorbars = chain(*errorbars)
        alpha = [x.get_alpha() for x in errorbars]
        # replace None with 1(None indicates default which is 1)
        alpha = [x if x is not None else 1 for x in alpha]
        if sum(alpha) == 0:
            has_errorbars = False
        else:
            has_errorbars = True
        return has_errorbars

    def _set_errorbars_shown_state(self, state):
        """Show errrorbar if state = 1, hide if state = 0"""
        current_axis = self.canvas.figure.gca()
        if state:
            alpha = 1
        else:
            alpha = 0.
        for i in range(len(current_axis.containers)):
            if isinstance(current_axis.containers[i], ErrorbarContainer):
                elements = current_axis.containers[i].get_children()
                if self.get_line_visible(i):
                    elements[1].set_alpha(alpha) #  elements[0] is the actual line, elements[1] is error bars


    def _toggle_errorbars(self):
        state = self._has_errorbars()
        if state is None:  # No errorbars in this plot
            return
        self._set_errorbars_shown_state(not state)

    def get_legends(self):
        current_axis = self.canvas.figure.gca()
        legends = []
        labels = current_axis.get_legend_handles_labels()[1]
        for i in range(len(labels)):
            try:
                v = self.legends_visible[i]
            except IndexError:
                v = True
                self.legends_visible.append(True)
            legends.append({'label': labels[i], 'visible': v})
        return legends

    def set_legends(self, legends):
        current_axes = self.canvas.figure.gca()
        if current_axes.legend_:
            current_axes.legend_.remove()  # remove old legends
        if legends is None or not self.legends_shown:
            return
        labels = []
        handles_to_show = []
        handles = current_axes.get_legend_handles_labels()[0]
        for i in range(len(legends)):
            container = current_axes.containers[i]
            container.set_label(legends[i]['label'])
            if legends[i]['visible']:
                handles_to_show.append(handles[i])
                labels.append(legends[i]['label'])
            self.legends_visible[i] = legends[i]['visible']
        x = current_axes.legend(handles_to_show, labels)  # add new legends
        x.draggable()

    def _toggle_legend(self):
        self.legends_shown = not self.legends_shown
        self.set_legends(self.get_legends())
        self.canvas.draw()

    def get_line_data(self):
        legends = self.get_legends()
        all_line_options = []
        i = 0
        for line_group in self.canvas.figure.gca().containers:
            line_options = {}
            line = line_group.get_children()[0]
            line_options['show'] = self.get_line_visible(i)
            line_options['color'] = line.get_color()
            line_options['style'] = line.get_linestyle()
            line_options['width'] = str(int(line.get_linewidth()))
            line_options['marker'] = line.get_marker()
            all_line_options.append(line_options)
            i += 1
        return list(zip(legends, all_line_options))

    def set_line_data(self, line_data):
        legends = []
        i = 0
        for line in line_data:
            legend, line_options = line
            legends.append(legend)
            line_model = self.canvas.figure.gca().containers[i]
            self.set_line_visible(i, line_options['show'])
            for child in line_model.get_children():
                child.set_color(line_options['color'])
                child.set_linewidth(line_options['width'])
            main_line = line_model.get_children()[0]
            main_line.set_linestyle(line_options['style'])
            main_line.set_marker(line_options['marker'])
            i += 1
        self.set_legends(legends)

    def set_line_visible(self, line_index, visible):
        self.lines_visible[line_index] = visible
        for child in self.canvas.figure.gca().containers[line_index].get_children():
            child.set_alpha(visible)

    def get_line_visible(self, line_index):
        try:
            ret = self.lines_visible[line_index]
            return ret
        except KeyError:
            self.lines_visible[line_index] = True
            return True

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
    def x_log(self):
        return 'log' in self.canvas.figure.gca().get_xscale()

    @property
    def y_log(self):
        return 'log' in self.canvas.figure.gca().get_yscale()

    @property
    def colorbar_range(self):
        return self.canvas.figure.gca().get_images()[0].get_clim()

    @property
    def colorbar_log(self):
        mappable = self.canvas.figure.gca().get_images()[0]
        norm = mappable.norm
        return isinstance(norm, colors.LogNorm)

    @property
    def error_bars(self):
        return self._has_errorbars()

    @error_bars.setter
    def error_bars(self, value):
        self._set_errorbars_shown_state(value)

    @property
    def show_legends(self):
        return self.legends_shown

    @show_legends.setter
    def show_legends(self, value):
        self.legends_shown = value
