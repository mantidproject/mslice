import PyQt4.QtGui as QtGui
from PyQt4.QtCore import pyqtSignal

from .plot_options_ui import Ui_Dialog


class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):

    titleChanged = pyqtSignal()

    def __init__(self):
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)
        self._legend_widgets = []
        self.lneFigureTitle.textEdited.connect(self._title_changed)

    def _title_changed(self):
        self.titleChanged.emit()

    def add_legend(self, text, handle, visible):
        legend_widget = LegendSetter(self, text, handle, visible)
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

    @x_range.setter
    def x_range(self, xrange):
        try:
            xmin, xmax = xrange
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
            return (None, None)
        return (ymin, ymax)

    @y_range.setter
    def y_range(self, yrange):
        try:
            ymin, ymax = yrange
        except ValueError:
            raise ValueError("pass an iterable with two items")
        self.lneYMin.setText(str(ymin))
        self.lneYMax.setText(str(ymax))

    @property
    def colorbar_range(self):
        try:
            cmin = float(str(self.lneCMin.text()))
            cmax = float(str(self.lneCMax.text()))
        except ValueError:
            return (None, None)
        return (cmin, cmax)

    @colorbar_range.setter
    def colorbar_range(self, c_range):
        try:
            cmin, cmax = c_range
        except ValueError:
            raise ValueError("pass an iterable with two items")
        self.lneCMin.setText(str(cmin))
        self.lneCMax.setText(str(cmax))

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


