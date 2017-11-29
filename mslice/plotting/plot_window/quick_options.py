
from mslice.plotting.plot_window.plot_options import LegendAndLineOptionsSetter

from PyQt4 import QtGui
from PyQt4.QtCore import pyqtSignal


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
