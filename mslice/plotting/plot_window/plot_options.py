from __future__ import (absolute_import, division, print_function)
import qtpy.QtWidgets as QtWidgets
from qtpy.QtCore import Signal
from six import iteritems

from .plot_options_ui import Ui_Dialog


class PlotOptionsDialog(QtWidgets.QDialog, Ui_Dialog):


    titleEdited = Signal()
    xLabelEdited = Signal()
    yLabelEdited = Signal()
    xRangeEdited = Signal()
    yRangeEdited = Signal()
    xGridEdited = Signal()
    yGridEdited = Signal()

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
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.chkXGrid.stateChanged.connect(self.xGridEdited)
        self.chkYGrid.stateChanged.connect(self.yGridEdited)

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

    @property
    def x_grid(self):
        return self.chkXGrid.isChecked()

    @x_grid.setter
    def x_grid(self, value):
        self.chkXGrid.setChecked(value)

    @property
    def y_grid(self):
        return self.chkYGrid.isChecked()

    @y_grid.setter
    def y_grid(self, value):
        self.chkYGrid.setChecked(value)



class SlicePlotOptions(PlotOptionsDialog):

    cRangeEdited = Signal()
    cLogEdited = Signal()

    def __init__(self):
        super(SlicePlotOptions, self).__init__()
        self.chkXLog.hide()
        self.chkYLog.hide()
        self.cut_options.hide()
        self.setMaximumWidth(350)

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

    xLogEdited = Signal()
    yLogEdited = Signal()
    showLegendsEdited = Signal()

    def __init__(self):
        super(CutPlotOptions, self).__init__()
        self._line_widgets = []
        self.groupBox_4.hide()

        self.chkXLog.stateChanged.connect(self.xLogEdited)
        self.chkYLog.stateChanged.connect(self.yLogEdited)
        self.chkShowLegends.stateChanged.connect(self.showLegendsEdited)

    def set_line_data(self, line_data):
        for line in line_data:
            line_widget = LegendAndLineOptionsSetter(line, self.color_validator)
            self.verticalLayout_legend.addWidget(line_widget)
            self._line_widgets.append(line_widget)

    def get_line_data(self):
        all_line_options = []
        for line_widget in self._line_widgets:
            line_options = {}
            for option in ['shown', 'color', 'style', 'width', 'marker', 'legend', 'label']:
                line_options[option] = getattr(line_widget, option)
            all_line_options.append(line_options)
        return all_line_options

    def color_validator(self, selected):
        count = 0
        for line_widget in self._line_widgets:
            if line_widget.get_color_index() == selected:
                count += 1
        if count <= 1:
            return True
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Selection Invalid")
        msg_box.setIcon(QtWidgets.QMessageBox.Warning)
        msg_box.setText("Cannot have two lines the same colour.")
        msg_box.exec_()
        return False

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


class LegendAndLineOptionsSetter(QtWidgets.QWidget):
    """This is a widget that has various legend and line controls for each line of a plot"""

    # dictionaries used to convert from matplotlib arguments to UI selection and vice versa
    colors = {'B': 'Blue', 'G': 'Green', 'R': 'Red', 'C': 'Cyan', 'M': 'Magenta', 'Y': 'Yellow',
                   'K': 'Black', 'W': 'White'}

    styles = {'-': 'Solid', '--': 'Dashed', '-.': 'Dashdot', ':': 'Dotted'}

    markers = {'o': 'Circle', ',': 'Pixel', '.': 'Point', 'v': 'Triangle down', '^': 'Triangle up',
                    '<': 'Triangle_left', '>': 'Triangle right', '1': 'Arrow down', '2': 'Arrow up',
                    '3': 'Arrow left', '4': 'Arrow right', '8': 'Octagon', 's': 'Square', 'p': 'Pentagon',
                    '*': 'Star', 'h': 'Hexagon 1', 'H': 'Hexagon 2', '+': 'Plus', 'x': 'X', 'D': 'Diamond',
                    'd': 'Diamond (thin)', '|': 'Vertical line', '_': 'Horizontal line', 'None': 'None'}

    inverse_colors = {v: k for k, v in iteritems(colors)}
    inverse_styles = {v: k for k, v in iteritems(styles)}
    inverse_markers = {v: k for k, v in iteritems(markers)}

    def __init__(self, line_options, color_validator):
        super(LegendAndLineOptionsSetter, self).__init__()
        self.legendText = QtWidgets.QLineEdit(self)
        self.legendText.setText(line_options['label'])
        self.color_validator = color_validator

        self.color_label = QtWidgets.QLabel(self)
        self.color_label.setText("Color:")
        self.line_color = QtWidgets.QComboBox(self)
        self.line_color.addItems(list(self.colors.values()))
        chosen_color_as_string = self.colors[line_options['color'].upper()]
        self.line_color.setCurrentIndex(self.line_color.findText(chosen_color_as_string))
        self.previous_color = self.line_color.currentIndex()

        self.style_label = QtWidgets.QLabel(self)
        self.style_label.setText("Style:")
        self.line_style = QtWidgets.QComboBox(self)
        self.line_style.addItems(list(self.styles.values()))
        chosen_style_as_string = self.styles[line_options['style']]
        self.line_style.setCurrentIndex(self.line_style.findText(chosen_style_as_string))

        self.width_label = QtWidgets.QLabel(self)
        self.width_label.setText("Width:")
        self.line_width = QtWidgets.QComboBox(self)
        self.line_width.addItems([str(x+1) for x in range(10)])
        self.line_width.setCurrentIndex(self.line_width.findText(line_options['width']))

        self.marker_label = QtWidgets.QLabel(self)
        self.marker_label.setText("Marker:")
        self.line_marker = QtWidgets.QComboBox(self)
        markers = list(self.markers.values())
        markers.sort()
        self.line_marker.addItems(markers)
        chosen_marker_as_string = self.markers[line_options['marker']]
        self.line_marker.setCurrentIndex(self.line_marker.findText(chosen_marker_as_string))

        self.show_line_label = QtWidgets.QLabel(self)
        self.show_line_label.setText("Show: ")
        self.show_line = QtWidgets.QCheckBox(self)
        self.show_line.setChecked(line_options['shown'])

        self.show_legend_label = QtWidgets.QLabel(self)
        self.show_legend_label.setText("Show legend: ")
        self.show_legend = QtWidgets.QCheckBox(self)
        self.show_legend.setChecked(line_options['legend'])
        self.show_legend.setEnabled(line_options['shown'])

        layout = QtWidgets.QVBoxLayout(self)
        row1 = QtWidgets.QHBoxLayout()
        layout.addLayout(row1)
        row2 = QtWidgets.QHBoxLayout()
        layout.addLayout(row2)
        row3 = QtWidgets.QHBoxLayout()
        layout.addLayout(row3)
        row4 = QtWidgets.QHBoxLayout()
        layout.addLayout(row4)
        layout.addStretch()

        row1.addWidget(self.show_legend)
        row1.addWidget(self.legendText)
        row2.addWidget(self.color_label)
        row2.addWidget(self.line_color)
        row2.addWidget(self.style_label)
        row2.addWidget(self.line_style)
        row3.addWidget(self.width_label)
        row3.addWidget(self.line_width)
        row3.addWidget(self.marker_label)
        row3.addWidget(self.line_marker)
        row4.addWidget(self.show_line_label)
        row4.addWidget(self.show_line)
        row4.addWidget(self.show_legend_label)
        row4.addWidget(self.show_legend)

        if self.color_validator is not None:
            self.line_color.currentIndexChanged.connect(lambda selected: self.color_valid(selected))
        self.show_line.stateChanged.connect(lambda state: self.show_line_changed(state))

    def color_valid(self, index):
        if self.color_validator is None:
            return
        if self.color_validator(index):
            self.previous_color = self.line_color.currentIndex()
        else:
            self.line_color.setCurrentIndex(self.previous_color)

    def show_line_changed(self, state):
        #  automatically shows/hides legend if line is shown/hidden
        self.show_legend.setEnabled(state)
        self.show_legend.setChecked(state)

    def get_color_index(self):
        return self.line_color.currentIndex()

    @property
    def legend(self):
        return self.show_legend.checkState()

    @property
    def label(self):
        return str(self.legendText.text())

    @property
    def shown(self):
        return bool(self.show_line.checkState())

    @property
    def color(self):
        return self.inverse_colors[str(self.line_color.currentText())]

    @property
    def style(self):
        return self.inverse_styles[str(self.line_style.currentText())]

    @property
    def width(self):
        return self.line_width.currentText()

    @property
    def marker(self):
        return self.inverse_markers[str(self.line_marker.currentText())]
