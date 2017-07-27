from itertools import chain

from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from matplotlib.container import ErrorbarContainer
import matplotlib.colors as colors
from PyQt4.QtCore import Qt
import numpy as np

from mslice.presenters.plot_options_presenter import PlotOptionsPresenter, PlotConfig, LegendDescriptor
from .plot_window_ui import Ui_MainWindow
from .base_qt_plot_window import BaseQtPlotWindow
from .plot_options import PlotOptionsDialog


class PlotFigureManager(BaseQtPlotWindow, Ui_MainWindow):
    def __init__(self, number, manager):
        super(PlotFigureManager, self).__init__(number, manager)

        self.actionKeep.triggered.connect(self._report_as_kept_to_manager)
        self.actionMakeCurrent.triggered.connect(self._report_as_current_to_manager)

        self.actionDump_To_Console.triggered.connect(self._dump_script_to_console)

        self.actionDataCursor.toggled.connect(self.toggle_data_cursor)
        self.stock_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.stock_toolbar.hide()

        self.actionZoom_In.triggered.connect(self.stock_toolbar.zoom)
        self.actionZoom_Out.triggered.connect(self.stock_toolbar.back)
        self.action_save_image.triggered.connect(self.stock_toolbar.save_figure)
        self.actionPlotOptions.triggered.connect(self._plot_options)
        self.actionToggleLegends.triggered.connect(self._toggle_legend)

        self.show()  # is not a good idea in non interactive mode

    def toggle_data_cursor(self):
        if self.actionDataCursor.isChecked():
            self.stock_toolbar.message.connect(self.statusbar.showMessage)
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self.stock_toolbar.message.disconnect()
            self.canvas.setCursor(Qt.ArrowCursor)

    def _display_status(self,status):
        if status == "kept":
            self.actionKeep.setChecked(True)
            self.actionMakeCurrent.setChecked(False)
        elif status == "current":
            self.actionMakeCurrent.setChecked(True)
            self.actionKeep.setChecked(False)

    def _plot_options(self):
        if self.canvas.figure.gca().get_images():
            color_plot = True
        else:
            color_plot = False
        view = PlotOptionsDialog(color_plot)
        new_config = PlotOptionsPresenter(view, self).get_new_config()
        if new_config:
            self.canvas.draw()

    def _get_min(self, data, absolute_minimum=-np.inf):
        """Determines the minimum of a set of numpy arrays"""
        data = data if isinstance(data, list) else [data]
        running_min = []
        for values in data:
            try:
                running_min.append(np.min(values[np.isfinite(values) * (values > absolute_minimum)]))
            except ValueError:  # If data is empty or not array of numbers
                pass
        return np.min(running_min) if running_min else absolute_minimum

    def set_legends(self, legends):
        current_axis = self.canvas.figure.gca()
        for handle in legends.handles:
            handle.set_label(legends.get_legend_text(handle))

        self.set_legend_state(legends.visible)

    def change_lineplot(self, xlog, ylog):
        current_axis = self.canvas.figure.gca()
        if xlog:
            xdata = [ll.get_xdata() for ll in current_axis.get_lines()]
            xmin = self._get_min(xdata, absolute_minimum=0.)
            current_axis.set_xscale('symlog', linthreshx=pow(10, np.floor(np.log10(xmin))))
        else:
            current_axis.set_xscale('linear')
        if ylog:
            ydata = [ll.get_ydata() for ll in current_axis.get_lines()]
            ymin = self._get_min(ydata, absolute_minimum=0.)
            current_axis.set_yscale('symlog', linthreshy=pow(10, np.floor(np.log10(ymin))))
        else:
            current_axis.set_yscale('linear')

    def change_colorplot(self, colorbar_range, logarithmic):
        current_axis = self.canvas.figure.gca()
        images = current_axis.get_images()
        if len(images) != 1:
            raise RuntimeError("Expected single image on axes, found " + str(len(images)))
        mappable = images[0]
        print current_axis.get_images
        mappable.set_clim(*colorbar_range)  # * unnecessary?
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

    def set_legend_state(self, visible=True):
        """Show legends if true, hide legends is visible is false"""
        current_axes = self.canvas.figure.gca()
        if visible:
            leg = current_axes.legend()
            leg.draggable()
        else:
            if current_axes.legend_:
                current_axes.legend_.remove()
                current_axes.legend_ = None

    def _toggle_legend(self):
        current_axes = self.canvas.figure.gca()
        if not list(current_axes._get_legend_handles()):
            return  # Legends are not applicable to this plot
        current_state = getattr(current_axes, 'legend_') is not None
        self.set_legend_state(not current_state)
        self.canvas.draw()

    def _has_errorbars(self):
        """True current axes has visible errorbars,
         False if current axes has hidden errorbars
         None if errorbars are not applicable"""
        current_axis = self.canvas.figure.gca()
        if not any(map(lambda x: isinstance(x, ErrorbarContainer),current_axis.containers)):
            has_errorbars = None  # Error bars are not applicable to this plot and will not show up in the config
        else:
            # If all the error bars have alpha= 0 they are all transparent (hidden)
            containers = filter(lambda x: isinstance(x, ErrorbarContainer), current_axis.containers)
            line_components = map(lambda x:x.get_children(), containers)
            # drop the first element of each container because it is the the actual line
            errorbars = map(lambda x: x[1:] , line_components)
            errorbars = chain(*errorbars)
            alpha = map(lambda x: x.get_alpha(), errorbars)
            # replace None with 1(None indicates default which is 1)
            alpha = map(lambda x: x if x is not None else 1, alpha)
            if sum(alpha) == 0:
                has_errorbars = False
            else:
                has_errorbars = True
        return has_errorbars

    def _set_errorbars_shown_state(self,state):
        """Show errrorbar if state = 1, hide if state = 0"""
        current_axis = self.canvas.figure.gca()
        if state:
            alpha = 1
        else:
            alpha = 0
        for container in current_axis.containers:
            if isinstance(container, ErrorbarContainer):
                elements = container.get_children()
                for i in range(len(elements)):
                    # The first component is the actual line so we will not touch it
                    if i != 0:
                        elements[i].set_alpha(alpha)

    def _toggle_errorbars(self):
        state = self._has_errorbars()
        if state is None: # No errorbars in this plot
            return
        self._set_errorbars_shown_state(not state)

    def get_title(self):
        return self.canvas.figure.gca().get_title()

    def get_x_label(self):
        return self.canvas.figure.gca().get_xlabel()

    def get_y_label(self):
        return self.canvas.figure.gca().get_ylabel()

    # if a legend has been set to '' or has been hidden (by prefixing with '_)then it will be ignored by
    # axes.get_legend_handles()
    # That is the reason for the use of the private function axes._get_legend_handles
    # This code was written against the 1.5.1 version of matplotlib.
    def get_legends(self):
        current_axis = self.canvas.figure.gca()
        handles = list(current_axis._get_legend_handles())
        if not handles:
            legend = LegendDescriptor(applicable=False)
        else:
            visible = getattr(current_axis, 'legend_') is not None
            legend = LegendDescriptor(visible=visible, handles=handles)
        return legend

