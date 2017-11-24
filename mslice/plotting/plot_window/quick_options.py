
from mslice.plotting.plot_window.plot_options import LegendAndLineOptionsSetter

from PyQt4 import QtGui


class QuickLineOptions(object):

    def __init__(self, line):
        line_options = {}
        line_options['show'] = True
        # print(target.get_color())
        # line_options['color'] = target.get_color()
        line_options['color'] = 'b'
        line_options['style'] = line.get_linestyle()
        line_options['width'] = str(int(line.get_linewidth()))
        line_options['marker'] = line.get_marker()
        dialog = QtGui.QDialog()
        dialog.setWindowTitle("Edit line")
        layout = QtGui.QVBoxLayout()
        dialog.setLayout(layout)
        self.line_widget = LegendAndLineOptionsSetter(dialog, line.get_label(), True, line_options)
        self.ok_button = QtGui.QPushButton("OK", dialog)
        self.cancel_button = QtGui.QPushButton("Cancel", dialog)
        new_row = QtGui.QHBoxLayout()
        layout.addWidget(self.line_widget)
        new_row.addWidget(self.ok_button)
        new_row.addWidget(self.cancel_button)
        layout.addLayout(new_row)
        self.line_widget.show()
        dialog.exec_()

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

