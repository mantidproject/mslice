
from mslice.plotting.plot_window.plot_options import LegendAndLineOptionsSetter

from mslice.util.qt import QtWidgets
from mslice.util.qt.QtCore import Signal


class QuickOptions(QtWidgets.QDialog):

    def __init__(self):
        super(QuickOptions, self).__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.ok_button = QtWidgets.QPushButton("OK", self)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.button_row = QtWidgets.QHBoxLayout()
        self.button_row.addWidget(self.ok_button)
        self.button_row.addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)


class QuickAxisOptions(QuickOptions):

    def __init__(self, target, existing_values, grid, log):
        super(QuickAxisOptions, self).__init__()
        self.setWindowTitle("Edit " + target)
        self.log = log
        self.min_label = QtWidgets.QLabel("Min:")
        self.min = QtWidgets.QLineEdit()
        self.min.setText(str(existing_values[0]))
        self.max_label = QtWidgets.QLabel("Max:")
        self.max = QtWidgets.QLineEdit()
        self.max.setText(str(existing_values[1]))
        row1 = QtWidgets.QHBoxLayout()
        row2 = QtWidgets.QHBoxLayout()
        row1.addWidget(self.min_label)
        row1.addWidget(self.min)
        row2.addWidget(self.max_label)
        row2.addWidget(self.max)
        self.layout.addLayout(row1)
        self.layout.addLayout(row2)
        if grid is not None:
            self.grid = QtWidgets.QCheckBox("Grid", self)
            self.grid.setChecked(grid)
            row3 = QtWidgets.QHBoxLayout()
            row3.addWidget(self.grid)
            self.layout.addLayout(row3)
        if log is not None:
            self.log_scale = QtWidgets.QCheckBox("Logarithmic", self)
            self.log_scale.setChecked(self.log)
            row4 = QtWidgets.QHBoxLayout()
            row4.addWidget(self.log_scale)
            self.layout.addLayout(row4)
        self.layout.addLayout(self.button_row)

    @property
    def range_min(self):
        return self.min.text()

    @property
    def range_max(self):
        return self.max.text()

    @property
    def grid_state(self):
        return self.grid.checkState()

class QuickLabelOptions(QuickOptions):

    ok_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self, label):
        super(QuickLabelOptions, self).__init__()
        self.setWindowTitle("Edit " + label.get_text())
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setText(label.get_text())
        self.layout.addWidget(self.line_edit)
        self.layout.addLayout(self.button_row)
        self.line_edit.show()

    @property
    def label(self):
        return self.line_edit.text()


class QuickLineOptions(QuickOptions):

    ok_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self, line_options):
        super(QuickLineOptions, self).__init__()
        self.setWindowTitle("Edit line")
        self.line_widget = LegendAndLineOptionsSetter(line_options, None)
        self.layout.addWidget(self.line_widget)
        self.layout.addLayout(self.button_row)

        self.line_widget.show()

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
