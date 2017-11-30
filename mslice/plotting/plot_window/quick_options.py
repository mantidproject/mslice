
from mslice.plotting.plot_window.plot_options import LegendAndLineOptionsSetter

from PyQt4 import QtGui
from PyQt4.QtCore import pyqtSignal


class QuickAxisOptions(QtGui.QDialog):

    ok_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, target, existing_values):
        super(QuickAxisOptions, self).__init__()
        self.setWindowTitle("Edit " + target)
        self.min_label = QtGui.QLabel("Min:")
        self.min = QtGui.QLineEdit()
        self.min.setText(str(existing_values[0]))
        self.max_label = QtGui.QLabel("Max:")
        self.max = QtGui.QLineEdit()
        self.max.setText(str(existing_values[1]))
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.ok_button = QtGui.QPushButton("OK", self)
        self.cancel_button = QtGui.QPushButton("Cancel", self)
        row1 = QtGui.QHBoxLayout()
        row2 = QtGui.QHBoxLayout()
        row3 = QtGui.QHBoxLayout()
        row1.addWidget(self.min_label)
        row1.addWidget(self.min)
        row2.addWidget(self.max_label)
        row2.addWidget(self.max)
        row3.addWidget(self.ok_button)
        row3.addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self.ok_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        self.show()

    @property
    def range_min(self):
        return self.min.text()

    @property
    def range_max(self):
        return self.max.text()

class QuickLabelOptions(QtGui.QDialog):

    ok_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, label):
        super(QuickLabelOptions, self).__init__()
        self.setWindowTitle("Edit " + label.get_text())
        self.line_edit = QtGui.QLineEdit()
        self.line_edit.setText(label.get_text())
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.line_edit)
        self.ok_button = QtGui.QPushButton("OK", self)
        self.cancel_button = QtGui.QPushButton("Cancel", self)
        new_row = QtGui.QHBoxLayout()
        new_row.addWidget(self.ok_button)
        new_row.addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self.ok_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked)
        layout.addLayout(new_row)
        self.line_edit.show()
        self.show()

    @property
    def label(self):
        return self.line_edit.text()


class QuickLineOptions(QtGui.QDialog):

    ok_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, line, parent=None):
        super(QuickLineOptions, self).__init__(parent)
        line_options = {}
        line_options['shown'] = True
        # print(target.get_color())
        # line_options['color'] = target.get_color()
        line_options['color'] = 'b'
        line_options['style'] = line.get_linestyle()
        line_options['width'] = str(int(line.get_linewidth()))
        line_options['marker'] = line.get_marker()

        self.setWindowTitle("Edit line")
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.line_widget = LegendAndLineOptionsSetter(line.get_label(), True, line_options, None)
        self.ok_button = QtGui.QPushButton("OK", self)
        self.cancel_button = QtGui.QPushButton("Cancel", self)
        new_row = QtGui.QHBoxLayout()
        layout.addWidget(self.line_widget)
        new_row.addWidget(self.ok_button)
        new_row.addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self.ok_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked)
        layout.addLayout(new_row)
        self.line_widget.show()
        self.show()

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
        return self.line_widget.get_text()

    @property
    def shown(self):
        return self.line_widget.shown

    @property
    def legend(self):
        return self.line_widget.legend_visible()
