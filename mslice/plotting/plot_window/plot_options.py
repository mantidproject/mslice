import PyQt4.QtGui as QtGui
from PyQt4.QtCore import pyqtSignal

from .plot_options_ui import Ui_Dialog


class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):

    titleEdited = pyqtSignal()
    xLabelEdited = pyqtSignal()
    yLabelEdited = pyqtSignal()
    xRangeEdited = pyqtSignal()
    yRangeEdited = pyqtSignal()

    def __init__(self):
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)

        self.lneFigureTitle.editingFinished.connect(self.titleEdited)
        self.lneXAxisLabel.editingFinished.connect(self.xLabelEdited)
        self.lneYAxisLabel.editingFinished.connect(self.yLabelEdited)
        self.lneXMin.editingFinished.connect(self.xRangeEdited)
        self.lneXMax.editingFinished.connect(self.xRangeEdited)
        self.lneYMin.editingFinished.connect(self.yRangeEdited)
        self.lneYMax.editingFinished.connect(self.yRangeEdited)

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

        self.lneCMin.editingFinished.connect(self.cRangeEdited)
        self.lneCMax.editingFinished.connect(self.cRangeEdited)
        self.chkLogarithmic.stateChanged.connect(self.cLogEdited)

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
    def colorbar_log(self):
        return self.chkLogarithmic.isChecked()

    @colorbar_log.setter
    def colorbar_log(self, value):
        self.chkLogarithmic.setChecked(value)


class CutPlotOptions(PlotOptionsDialog):

    xLogEdited = pyqtSignal()
    yLogEdited = pyqtSignal()
    errorBarsEdited = pyqtSignal()
    showLegendsEdited = pyqtSignal()

    def __init__(self):
        super(CutPlotOptions, self).__init__()
        self._line_widgets = []
        self.groupBox_4.hide()

        self.chkXLog.stateChanged.connect(self.xLogEdited)
        self.chkYLog.stateChanged.connect(self.yLogEdited)
        self.chkShowErrorBars.stateChanged.connect(self.errorBarsEdited)
        self.chkShowLegends.stateChanged.connect(self.showLegendsEdited)

    def set_line_data(self, line_data):
        for line in line_data:
            legend, line_options = line
            line_widget = LegendAndLineOptionsSetter(self, legend['label'], legend['visible'], line_options)
            self.verticalLayout.addWidget(line_widget)
            self._line_widgets.append(line_widget)

    def get_legends(self):
        legends = []
        for line_widget in self._line_widgets:
            legends.append({'label': line_widget.get_text(), 'visible': line_widget.is_visible()})
        return legends

    def get_line_data(self):
        legends = self.get_legends()
        all_line_options = []
        for line_widget in self._line_widgets:
            line_options = {}
            for option in ['color', 'style', 'width', 'marker']:
                line_options[option] = getattr(line_widget, option)
            all_line_options.append(line_options)
        return zip(legends, all_line_options)

    @property
    def x_log(self):
        return self.chkXLog.isChecked()

    @x_log.setter
    def x_log(self, value):
        self.chkXLog.setChecked(value)

    @property
    def y_log(self):
        return self.chkYLog.isChecked()

    @y_log.setter
    def y_log(self, value):
        self.chkYLog.setChecked(value)

    @property
    def error_bars(self):
        return self.chkShowErrorBars.isChecked()

    @error_bars.setter
    def error_bars(self, value):
        self.chkShowErrorBars.setChecked(value)

    @property
    def show_legends(self):
        return self.chkShowLegends.isChecked()

    @show_legends.setter
    def show_legends(self, value):
        self.chkShowLegends.setChecked(value)


class LegendAndLineOptionsSetter(QtGui.QWidget):
    """This is a widget that has various legend and line controls for each line of a plot
    This widget has a concrete reference to the artist and modifies it"""
    def __init__(self, parent, text, is_enabled, line_options):
        super(LegendAndLineOptionsSetter, self).__init__(parent)
        self.isEnabled = QtGui.QCheckBox(self)
        self.isEnabled.setChecked(is_enabled)
        self.legendText = QtGui.QLineEdit(self)
        self.legendText.setText(text)

        self.color_label = QtGui.QLabel(self)
        self.color_label.setText("Color:")
        self.line_color = QtGui.QComboBox(self)
        self.line_color.addItems(['b', 'g', 'r', 'c', 'm', 'y', 'b', 'w'])
        self.line_color.setCurrentIndex(self.line_color.findText(line_options['color']))

        self.style_label = QtGui.QLabel(self)
        self.style_label.setText("Style:")
        self.line_style = QtGui.QComboBox(self)
        self.line_style.addItems(['-', '--', '-.', ':'])
        self.line_style.setCurrentIndex(self.line_style.findText(line_options['style']))

        self.width_label = QtGui.QLabel(self)
        self.width_label.setText("Width:")
        self.line_width = QtGui.QComboBox(self)
        self.line_width.addItems(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        self.line_width.setCurrentIndex(self.line_width.findText(line_options['width']))

        self.marker_label = QtGui.QLabel(self)
        self.marker_label.setText("Marker:")
        self.line_marker = QtGui.QComboBox(self)
        self.line_marker.addItems(['o', ',', '.', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p' 'P',
                                   '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 'None'])
        self.line_marker.setCurrentIndex(self.line_marker.findText(line_options['marker']))

        layout = QtGui.QVBoxLayout(self)
        row1 = QtGui.QHBoxLayout()
        layout.addLayout(row1)
        row2 = QtGui.QHBoxLayout()
        layout.addLayout(row2)
        row3 = QtGui.QHBoxLayout()
        layout.addLayout(row3)

        row1.addWidget(self.isEnabled)
        row1.addWidget(self.legendText)
        row2.addWidget(self.color_label)
        row2.addWidget(self.line_color)
        row2.addWidget(self.style_label)
        row2.addWidget(self.line_style)
        row3.addWidget(self.width_label)
        row3.addWidget(self.line_width)
        row3.addWidget(self.marker_label)
        row3.addWidget(self.line_marker)

    def is_visible(self):
        return self.isEnabled.checkState()

    def get_text(self):
        return str(self.legendText.text())

    @property
    def color(self):
        return self.line_color.currentText()

    @property
    def style(self):
        return self.line_style.currentText()

    @property
    def width(self):
        return self.line_width.currentText()

    @property
    def marker(self):
        return self.line_marker.currentText()
