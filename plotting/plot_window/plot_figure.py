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
        self.canvas.draw()

    def _get_plot_description(self):
        title = self.canvas.figure.gca().get_title()
        xlabel = self.canvas.figure.gca().get_xlabel()
        ylabel = self.canvas.figure.gca().get_ylabel()
        return PlotConfig(title=title, x_axis_label=xlabel, y_axis_label=ylabel)

class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):
    def __init__(self, current_config):
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)
        self.groupBox.hide()
        self.chkShowErrorBars.hide()
        self.lneFigureTitle.setText(current_config.title)
        self.lneXAxisLabel.setText(current_config.xlabel)
        self.lneYAxisLabel.setText(current_config.ylabel)


    @staticmethod
    def get_new_config(current_config):
        dialog = PlotOptionsDialog(current_config)
        dialog.exec_()
        return PlotConfig(title=dialog.lneFigureTitle.text(),
                          x_axis_label=dialog.lneXAxisLabel.text(),
                          y_axis_label=dialog.lneYAxisLabel.text())


class PlotConfig(object):
    def __init__(self, title=None, x_axis_label=None, y_axis_label=None, legends=None, errorbars_enabled=None):
        self.title = title
        self.xlabel = x_axis_label
        self.ylabel = y_axis_label
        self._legends = legends
        self._errorbar = errorbars_enabled   # Has 3 values (True : shown, False: Not Shown, None: Not applicable)

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

