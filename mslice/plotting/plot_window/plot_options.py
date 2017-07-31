import PyQt4.QtGui as QtGui
from abc import ABCMeta
from PyQt4.QtCore import pyqtSignal

from .plot_options_ui import Ui_Dialog
from mslice.presenters.plot_options_presenter import LegendDescriptor


class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):

    titleEdited = pyqtSignal()
    xLabelEdited = pyqtSignal()
    yLabelEdited = pyqtSignal()
    xRangeEdited = pyqtSignal()
    yRangeEdited = pyqtSignal()

    def __init__(self):
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)

        self.lneFigureTitle.editingFinished.connect(self._title_edited)
        self.lneXAxisLabel.editingFinished.connect(self._x_label_edited)
        self.lneYAxisLabel.editingFinished.connect(self._y_label_edited)
        self.lneXMin.editingFinished.connect(self._x_range_edited)
        self.lneXMax.editingFinished.connect(self._x_range_edited)
        self.lneYMin.editingFinished.connect(self._y_range_edited)
        self.lneYMax.editingFinished.connect(self._y_range_edited)

    def _title_edited(self):
        self.titleEdited.emit()

    def _x_label_edited(self):
        self.xLabelEdited.emit()

    def _y_label_edited(self):
        self.yLabelEdited.emit()

    def _x_range_edited(self):
        self.xRangeEdited.emit()

    def _y_range_edited(self):
        self.yRangeEdited.emit()

    @property
    def x_range(self):
        try:
            xmin = float(str(self.lneXMin.text()))
            xmax = float(str(self.lneXMax.text()))
        except ValueError:
            return None, None
        return xmin, xmax

    @x_range.setter
    def x_range(self, x_range):
        try:
            xmin, xmax = x_range
        except ValueError:
            raise ValueError("pass an iterable with two items")
        self.lneXMin.setText(str(xmin))
        self.lneXMax.setText(str(xmax))

    @property
    def y_range(self):
        try:
            ymin = float(str(self.lneYMin.text()))
            ymax = float(str(self.lneYMax.text()))
        except ValueError:
            return None, None
        return ymin, ymax

    @y_range.setter
    def y_range(self, yrange):
        try:
            ymin, ymax = yrange
        except ValueError:
            raise ValueError("pass an iterable with two items")
        self.lneYMin.setText(str(ymin))
        self.lneYMax.setText(str(ymax))

    @property
    def title(self):
        return self.lneFigureTitle.text()

    @title.setter
    def title(self, value):
        self.lneFigureTitle.setText(value)

    @property
    def x_label(self):
        return self.lneXAxisLabel.text()

    @x_label.setter
    def x_label(self, value):
        self.lneXAxisLabel.setText(value)

    @property
    def y_label(self):
        return self.lneYAxisLabel.text()

    @y_label.setter
    def y_label(self, value):
        self.lneYAxisLabel.setText(value)


class SlicePlotOptions(PlotOptionsDialog):

    cRangeEdited = pyqtSignal()
    cLogEdited = pyqtSignal()

    def __init__(self):
        super(SlicePlotOptions, self).__init__()
        self.chkXLog.hide()
        self.chkYLog.hide()
        self.chkShowErrorBars.hide()
        self.groupBox.hide()

        self.lneCMin.editingFinished.connect(self._c_range_edited)
        self.lneCMax.editingFinished.connect(self._c_range_edited)
        self.chkLogarithmic.stateChanged.connect(self._c_log_edited)

    def _c_range_edited(self):
        self.cRangeEdited.emit()

    def _c_log_edited(self):
        self.cLogEdited.emit()

    def set_log(self, value):
            self.chkLogarithmic.setChecked(value)

    @property
    def colorbar_range(self):
        try:
            cmin = float(str(self.lneCMin.text()))
            cmax = float(str(self.lneCMax.text()))
        except ValueError:
            return None, None
        return cmin, cmax

    @colorbar_range.setter
    def colorbar_range(self, c_range):
        try:
            cmin, cmax = c_range
        except ValueError:
            raise ValueError("pass an iterable with two items")
        self.lneCMin.setText(str(cmin))
        self.lneCMax.setText(str(cmax))

    @property
    def c_log(self):
        return self.chkLogarithmic.isChecked()


class CutPlotOptions(PlotOptionsDialog):

    xLogEdited = pyqtSignal()
    yLogEdited = pyqtSignal()
    errorBarsEdited = pyqtSignal()

    def __init__(self):
        super(CutPlotOptions, self).__init__()
        self._legend_widgets = []
        self.groupBox_4.hide()

        self.chkXLog.stateChanged.connect(self._x_log_edited)
        self.chkYLog.stateChanged.connect(self._y_log_edited)
        self.chkShowErrorBars.stateChanged.connect(self._error_bars_edited)

    def set_legends(self, legends):
        if not legends.applicable:
            self.groupBox.hide()
        else:
            self.chkShowLegends.setChecked(legends.visible)
            for legend in legends.all_legends():
                self.add_legend(legend['text'], legend['handle'], legend['visible'])

    def get_legends(self):
        legends = LegendDescriptor(visible=self.chkShowLegends.isChecked(),
                                   applicable=self.groupBox.isHidden())
        for legend_widget in self._legend_widgets:
            legends.set_legend_text(handle=legend_widget.handle,
                                    text=legend_widget.get_text(),
                                    visible=legend_widget.is_visible())
        return legends

    def add_legend(self, text, handle, visible):
        legend_widget = LegendSetter(self, text, handle, visible)
        self.verticalLayout.addWidget(legend_widget)
        self._legend_widgets.append(legend_widget)

    def _x_log_edited(self):
        self.xLogEdited.emit()

    def _y_log_edited(self):
        self.yLogEdited.emit()

    def _error_bars_edited(self):
        self.errorBarsEdited.emit()

    def set_show_error_bars(self, value):
        if value is not None:
            self.chkShowErrorBars.setChecked(value)

    def set_log(self, to_set, value):
        if to_set == 'x':
            self.chkXLog.setChecked(value)
        elif to_set == 'y':
            self.chkYLog.setChecked(value)
        else:
            raise ValueError("must specify whether to set x or y")

    @property
    def x_log(self):
        return self.chkXLog.isChecked()

    @property
    def y_log(self):
        return self.chkYLog.isChecked()

    @property
    def error_bars(self):
        return self.chkShowErrorBars.isChecked()


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
