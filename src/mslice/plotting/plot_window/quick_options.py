from mslice.plotting.plot_window.plot_options import LegendAndLineOptionsSetter

from mantidqt.utils.qt.line_edit_double_validator import LineEditDoubleValidator

from qtpy import QtWidgets
from qtpy.QtCore import Signal


class QuickOptions(QtWidgets.QDialog):

    ok_clicked = Signal()

    def __init__(self, parent=None):
        super(QuickOptions, self).__init__(parent)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.ok_button = QtWidgets.QPushButton("OK", self)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.button_row = QtWidgets.QHBoxLayout()
        self.button_row.addWidget(self.ok_button)
        self.button_row.addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.keep_open = QtWidgets.QCheckBox('Keep open')


class QuickAxisOptions(QuickOptions):

    def __init__(self, parent, target, existing_values, font_size, grid, log, redraw_signal):
        super(QuickAxisOptions, self).__init__(parent)
        self.setWindowTitle("Edit " + target)
        self.log = log
        self.min_label = QtWidgets.QLabel("Min:")
        self.min = QtWidgets.QLineEdit()
        self.min.setText(str(existing_values[0]))
        self.min_validator = LineEditDoubleValidator(self.min, str(existing_values[0]))
        self.min.setValidator(self.min_validator)

        self.max_label = QtWidgets.QLabel("Max:")
        self.max = QtWidgets.QLineEdit()
        self.max.setText(str(existing_values[1]))
        self.max_validator = LineEditDoubleValidator(self.max, str(existing_values[1]))
        self.max.setValidator(self.max_validator)

        self.font_size_label = QtWidgets.QLabel("Font Size:")
        self.font_size = QtWidgets.QDoubleSpinBox()
        self.decimals = 1
        self.font_size.setValue(font_size)
        self.all_fonts_size = QtWidgets.QCheckBox("Update All Fonts")

        row1 = QtWidgets.QHBoxLayout()
        row2 = QtWidgets.QHBoxLayout()
        row3 = QtWidgets.QHBoxLayout()
        row1.addWidget(self.min_label)
        row1.addWidget(self.min)
        row2.addWidget(self.max_label)
        row2.addWidget(self.max)
        row3.addWidget(self.font_size_label)
        row3.addWidget(self.font_size)
        self.layout.addLayout(row1)
        self.layout.addLayout(row2)
        self.layout.addLayout(row3)
        if grid is not None:
            self.grid = QtWidgets.QCheckBox("Grid", self)
            self.grid.setChecked(grid)
            row4 = QtWidgets.QHBoxLayout()
            row4.addWidget(self.grid)
            self.layout.addLayout(row4)
        if log is not None:
            self.log_scale = QtWidgets.QCheckBox("Logarithmic", self)
            self.log_scale.setChecked(self.log)
            row5 = QtWidgets.QHBoxLayout()
            row5.addWidget(self.log_scale)
            self.layout.addLayout(row5)
        self.layout.addWidget(self.all_fonts_size)   # Tick box for all font sizes
        self.layout.addLayout(self.button_row)
        self.layout.addWidget(self.keep_open)
        self.ok_button.clicked.disconnect()
        self.ok_button.clicked.connect(self._ok_clicked)
        self.redraw_signal = redraw_signal

    def _ok_clicked(self):
        self.ok_clicked.emit()
        self.redraw_signal.emit()
        if not self.is_kept_open:
            self.accept()

    @property
    def range_min(self):
        return self.min.text()

    @property
    def range_max(self):
        return self.max.text()

    @property
    def grid_state(self):
        return self.grid.checkState()

    @property
    def is_kept_open(self):
        return self.keep_open.isChecked()


class QuickLabelOptions(QuickOptions):

    ok_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self, parent, label, redraw_signal):
        super(QuickLabelOptions, self).__init__(parent)
        self.setWindowTitle("Edit " + label.get_text())
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setText(label.get_text())
        self.font_size_label = QtWidgets.QLabel("Font Size:")
        self.font_size = QtWidgets.QDoubleSpinBox()
        self.font_size.setValue(label.get_size())

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(self.font_size_label)
        row1.addWidget(self.font_size)

        self.layout.addWidget(self.line_edit)
        self.layout.addLayout(row1)
        self.layout.addLayout(self.button_row)
        self.line_edit.show()

        self.redraw_signal = redraw_signal
        self.ok_button.disconnect()
        self.ok_button.clicked.connect(self._ok_clicked)

    @property
    def label(self):
        return self.line_edit.text()

    @property
    def label_font_size(self):
        return self.font_size.value()

    def _ok_clicked(self):
        self.ok_clicked.emit()
        self.redraw_signal.emit()
        self.accept()


class QuickLineOptions(QuickOptions):

    ok_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self, parent, line_options, show_legends):
        super(QuickLineOptions, self).__init__(parent)
        self.setWindowTitle("Edit line")
        self.line_widget = LegendAndLineOptionsSetter(line_options, None, show_legends)
        self.layout.addWidget(self.line_widget)
        self.layout.addLayout(self.button_row)

        self.line_widget.show()

    @property
    def error_bar(self):
        return self.line_widget.error_bar

    @property
    def color(self):
        return self.line_widget.color

    @property
    def style(self):
        return self.line_widget.style

    @property
    def marker(self):
        return self.line_widget.marker

    @property
    def width(self):
        return self.line_widget.width

    @property
    def label(self):
        return self.line_widget.label

    @property
    def shown(self):
        return self.line_widget.shown

    @property
    def legend(self):
        return self.line_widget.legend


def QuickError(message):
    error_box = QtWidgets.QMessageBox()
    error_box.setWindowTitle("Error")
    error_box.setIcon(QtWidgets.QMessageBox.Warning)
    error_box.setText(message)
    error_box.exec_()
