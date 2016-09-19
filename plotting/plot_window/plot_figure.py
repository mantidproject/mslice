from base_qt_plot_window import BaseQtPlotWindow
from plotting.plot_window.plot_window_ui import Ui_MainWindow
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from PyQt4.QtCore import Qt
import PyQt4.QtGui as QtGui
from plot_options_ui import Ui_Dialog
from matplotlib.container import ErrorbarContainer
from itertools import chain


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
        config = self._get_plot_description()
        new_config = PlotOptionsDialog.get_new_config(config)
        self._apply_config(new_config)

    def _apply_config(self, plot_config):
        current_axis = self.canvas.figure.gca()
        current_axis.set_title(plot_config.title)
        current_axis.set_xlabel(plot_config.xlabel)
        current_axis.set_ylabel(plot_config.ylabel)
        legend_config = plot_config.legend
        for handle in legend_config.handles:
            handle.set_label(legend_config.get_legend_text(handle))

        # To show/hide errorbars we will just set the alpha to 0
        if plot_config.errorbar is not None:
            if plot_config.errorbar:
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

        # The legend must be set after hiding/showing the error bars so the errorbars on the legend are in sync with
        # the plot (in terms of having/not having errorbars)
        if legend_config.visible:
            current_axis.legend()

        else:
            if current_axis.legend_:
                current_axis.legend_.remove()
            current_axis.legend_ = None

        self.canvas.draw()

    def _get_plot_description(self):
        current_axis = self.canvas.figure.gca()
        title = current_axis.get_title()
        xlabel = current_axis.get_xlabel()
        ylabel = current_axis.get_ylabel()
        # if a legend has been set to '' or has been hidden (by prefixing with '_)then it will be ignored by
        # axes.get_legend_handles_labels()
        # That is the reason for the use of the private function axes._get_legend_handles
        # This code was written against the 1.5.1 version of matplotlib.
        handles = list(current_axis._get_legend_handles())
        labels = map(lambda x: x.get_label(), handles)
        labels = list(labels)
        if not handles:
            legend = LegendDescriptor(applicable=False)
        else:
            visible = getattr(current_axis, 'legend_') is not None
            legend = LegendDescriptor(visible=visible, handles=handles)
        if not any(map(lambda x: isinstance(x, ErrorbarContainer),current_axis.containers)):
            has_errorbars = None  # Error bars are not applicable to this plot and will not show up in the config
        else:
            # If all the error bars have alpha= 0 they are all transparent (hidden)
            containers = filter(lambda x: isinstance(x, ErrorbarContainer), current_axis.containers)
            line_components = map(lambda x:x.get_children(), containers)
            # drop the first element of each container because it is the the actual line
            errorbars = map(lambda x: x[1:] , line_components)
            errorbars =chain(*errorbars)
            alpha = map(lambda x: x.get_alpha(), errorbars)
            # replace None with 1(None indicates default which is 1)
            alpha = map(lambda x: x if x is not None else 1, alpha)
            if sum(alpha) == 0:
                has_errorbars = False
            else:
                has_errorbars = True
        return PlotConfig(title=title, x_axis_label=xlabel, y_axis_label=ylabel, legends=legend,
                          errorbars_enabled=has_errorbars)


class LegendSetter(QtGui.QWidget):
    """This is a widget that consists of a checkbox and a lineEdit that will control exactly one legend entry

    This widget has a concrete reference to the artist and modifies it"""
    def __init__(self, parent, text, handle, is_enabled):
        super(LegendSetter, self).__init__(parent)
        self.isEnabled = QtGui.QCheckBox(self)
        self.isEnabled.setChecked(is_enabled)
        self.legendText = QtGui.QLineEdit(self)
        self.legendText.setText(text)
        self.handle = handle
        layout = QtGui.QHBoxLayout(self)
        layout.addWidget(self.isEnabled)
        layout.addWidget(self.legendText)

    def is_visible(self):
        return self.isEnabled.checkState()

    def get_text(self):
        return str(self.legendText.text())



class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):
    def __init__(self, current_config):
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)
        self.lneFigureTitle.setText(current_config.title)
        self.lneXAxisLabel.setText(current_config.xlabel)
        self.lneYAxisLabel.setText(current_config.ylabel)
        self._legend_widgets = []
        self.chkShowLegends.setChecked(current_config.legend.visible)
        if current_config.errorbar is None:
            self.chkShowErrorBars.hide()
        else:
            self.chkShowErrorBars.setChecked(current_config.errorbar)
        if not current_config.legend.applicable:
            self.groupBox.hide()
        else:
            self.chkShowLegends.setChecked(current_config.legend.visible)
            for legend in current_config.legend.all_legends():
                legend_widget = LegendSetter(self, legend['text'], legend['handle'], legend['visible'])
                self.verticalLayout.addWidget(legend_widget)
                self._legend_widgets.append(legend_widget)

    def process_legends(self):
        for widget in self._legend_widgets:
            widget.apply_to_handle()

    @staticmethod
    def get_new_config(current_config):
        dialog = PlotOptionsDialog(current_config)
        dialog.exec_()
        legends = LegendDescriptor(visible=dialog.chkShowLegends.isChecked(),
                                   applicable=dialog.groupBox.isHidden())
        for legend_widget in dialog._legend_widgets:
            legends.set_legend_text(handle=legend_widget.handle,
                                    text=legend_widget.get_text(),
                                    visible=legend_widget.is_visible())

        return PlotConfig(title=dialog.lneFigureTitle.text(),
                          x_axis_label=dialog.lneXAxisLabel.text(),
                          y_axis_label=dialog.lneYAxisLabel.text(),
                          legends=legends,
                          errorbars_enabled=dialog.chkShowErrorBars.isChecked())


class LegendDescriptor(object):
    def __init__(self, visible=False, applicable=True, handles=None):
        self.visible = visible
        self.applicable = applicable
        if handles:
            self.handles = list(handles)
        else:
            self.handles = []
        self._labels = {}

    def all_legends(self):
        """An iterator which yields a dictionary description of legends containing the handle, text and if visible or not"""
        for handle in self.handles:
            yield  self.get_legend_descriptor(handle)

    def set_legend_text(self, handle, text, visible=True):
        if handle not in self.handles:
            self.handles.append(handle)
        if not visible:
            text = '_' + text
        self._labels[handle] = text

    def get_legend_descriptor(self, handle):
            if handle in self._labels.keys():
                label = self._labels[handle]  # If a new value has been set for a handle return that
            else:
                label = handle.get_label()   # Else get the value from the plot
            if label.startswith('_'):
                x = {'text': label[1:], 'visible': False, 'handle':handle}
            else:
                x = {'text': label, 'visible': True, 'handle':handle}
            return x

    def get_legend_text(self, handle):
        if handle in self._labels.keys():
            return self._labels[handle]
        return handle.get_label()


class PlotConfig(object):
    def __init__(self, title=None, x_axis_label=None, y_axis_label=None, legends=None, errorbars_enabled=None):
        self.title = title
        self.xlabel = x_axis_label
        self.ylabel = y_axis_label
        if legends is None:
            self.legend = LegendDescriptor()
        else:
            self.legend = legends
        self.errorbar = errorbars_enabled   # Has 3 values (True : shown, False: Not Shown, None: Not applicable)

    @property
    def title(self):
        if self._title is not None:
            return self._title
        return ""

    @title.setter
    def title(self, value):
        if value is None:
            self._title = None
        else:
            try:
                self._title = str(value)
            except ValueError:
                raise ValueError("Plot title must be a string or castable to string")

    @property
    def xlabel(self):
        if self._xlabel is not None:
            return self._xlabel
        return ""

    @xlabel.setter
    def xlabel(self, value):
        if value is None:
            self._xlabel = None
        else:
            try:
                self._xlabel = str(value)
            except ValueError:
                raise ValueError("Plot xlabel must be a string or castable to string")

    @property
    def ylabel(self):
        if self._ylabel is not None:
            return self._ylabel
        return ""

    @ylabel.setter
    def ylabel(self, value):
        if value is None:
            self._ylabel = None
        else:
            try:
                self._ylabel = str(value)
            except ValueError:
                raise ValueError("Plot ylabel must be a string or castable to string")

