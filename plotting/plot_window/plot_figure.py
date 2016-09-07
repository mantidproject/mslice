from base_qt_plot_window import BaseQtPlotWindow
from plotting.plot_window.plot_window_ui import Ui_MainWindow
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT
from PyQt4.QtCore import Qt
import PyQt4.QtGui as QtGui
from plot_options_ui import Ui_Dialog


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

        self.show() # is not a good idea in non interactive mode


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
        if plot_config.legend.visible:
            current_axis.legend(plot_config.legend.handles, plot_config.legend.labels)
        else:
            current_axis.legend_ = None
        self.canvas.draw()

    def _get_plot_description(self):
        current_axis = self.canvas.figure.gca()
        title = current_axis.get_title()
        xlabel = current_axis.get_xlabel()
        ylabel = current_axis.get_ylabel()
        handles, labels = current_axis.get_legend_handles_labels()
        if not handles:
            legend = LegendDescriptor(applicable=False)
        else:
            visible = hasattr(current_axis, 'legend_')
            legend = LegendDescriptor(visible=visible, handles=handles, labels=labels)

        return PlotConfig(title=title, x_axis_label=xlabel, y_axis_label=ylabel, legends=legend)


class LegendSetter(QtGui.QWidget):
    """This is a widget that consists of a checkbox and a lineEdit that will control exactly one legend entry"""
    def __init__(self, parent, text, is_enabled):
        super(LegendSetter, self).__init__(parent)
        self.isEnabled = QtGui.QCheckBox(self)
        self.isEnabled.setChecked(is_enabled)
        self.legendText = QtGui.QLineEdit(self)
        self.legendText.setText(text)

        layout = QtGui.QHBoxLayout(self)
        layout.addWidget(self.isEnabled)
        layout.addWidget(self.legendText)


class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):
    def __init__(self, current_config):
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)
        self.chkShowErrorBars.hide()
        self.lneFigureTitle.setText(current_config.title)
        self.lneXAxisLabel.setText(current_config.xlabel)
        self.lneYAxisLabel.setText(current_config.ylabel)
        if not current_config.legend.applicable:
            self.groupBox.hide()
        else:
            self.chkShowLegends.setChecked(current_config.legend.visible)
            for legend in current_config.legend.all_legends():
                self.verticalLayout.addWidget(LegendSetter(self, legend['text'], legend['enabled']))

    @staticmethod
    def get_new_config(current_config):
        dialog = PlotOptionsDialog(current_config)
        dialog.exec_()
        legends = LegendDescriptor(visible=dialog.chkShowLegends.isChecked())
        return PlotConfig(title=dialog.lneFigureTitle.text(),
                          x_axis_label=dialog.lneXAxisLabel.text(),
                          y_axis_label=dialog.lneYAxisLabel.text(),
                          legends=legends)


class LegendDescriptor(object):
    def __init__(self, visible=False, applicable=True, handles=None, labels=None):
        self.visible = visible
        self.applicable = applicable
        self.handles = handles
        self.labels = labels

    def all_legends(self):
        #TODO THIS IS JUST PLAIN WRONG
        for label in self.labels:
            x = {'text': label, 'enabled': True}
            yield x

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

