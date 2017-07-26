import PyQt4.QtGui as QtGui

from .plot_options_ui import Ui_Dialog


class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):
    def __init__(self, current_config): # noqa: C901
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)
        if current_config.title is not None:
            self.lneFigureTitle.setText(current_config.title)
        if current_config.xlabel is not None:
            self.lneXAxisLabel.setText(current_config.xlabel)
        if current_config.ylabel is not None:
            self.lneYAxisLabel.setText(current_config.ylabel)
        if None not in current_config.x_range:
            self.lneXMin.setText(str(current_config.x_range[0]))
            self.lneXMax.setText(str(current_config.x_range[1]))
        if None not in current_config.y_range:
            self.lneYMin.setText(str(current_config.y_range[0]))
            self.lneYMax.setText(str(current_config.y_range[1]))
        if None not in current_config.colorbar_range:
            self.lneCMin.setText(str(current_config.colorbar_range[0]))
            self.lneCMax.setText(str(current_config.colorbar_range[1]))
            self.chkXLog.hide()
            self.chkYLog.hide()
        else:
            self.groupBox_4.hide()
        if current_config.logarithmic is not None:
            self.chkLogarithmic.setChecked(current_config.logarithmic)
        if current_config.xlog is not None:
            self.chkXLog.setChecked(current_config.xlog)
        if current_config.ylog is not None:
            self.chkYLog.setChecked(current_config.ylog)

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

    @property
    def x_range(self):
        try:
            xmin = float(str(self.lneXMin.text()))
            xmax = float(str(self.lneXMax.text()))
        except ValueError:
            return (None, None)
        return (xmin, xmax)

    @property
    def y_range(self):
        try:
            ymin = float(str(self.lneYMin.text()))
            ymax = float(str(self.lneYMax.text()))
        except ValueError:
            return (None, None)
        return (ymin, ymax)

    @property
    def colorbar_range(self):
        try:
            cmin = float(str(self.lneCMin.text()))
            cmax = float(str(self.lneCMax.text()))
        except ValueError:
            return (None, None)
        return (cmin, cmax)


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


