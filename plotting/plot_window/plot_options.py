import PyQt4.QtGui as QtGui
from plot_options_ui import Ui_Dialog


class PlotOptionsDialog(QtGui.QDialog, Ui_Dialog):
    def __init__(self, current_config):
        super(PlotOptionsDialog, self).__init__()
        self.setupUi(self)
        self.lneFigureTitle.setText(current_config.title)
        self.lneXAxisLabel.setText(current_config.xlabel)
        self.lneYAxisLabel.setText(current_config.ylabel)
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
        if None not in current_config.x_range:
            self.lneXMin.setText(str(current_config.x_range[0]))
            self.lneXMax.setText(str(current_config.x_range[1]))
        if None not in current_config.y_range:
            self.lneYMin.setText(str(current_config.y_range[0]))
            self.lneYMax.setText(str(current_config.y_range[1]))
        if None not in current_config.colorbar_range:
            self.lneCMin.setText(str(current_config.colorbar_range[0]))
            self.lneCMax.setText(str(current_config.colorbar_range[1]))

    @staticmethod
    def get_new_config(current_config):
        dialog = PlotOptionsDialog(current_config)
        dialog_accepted=dialog.exec_()
        if not dialog_accepted:
            return None
        try:
            xmin = float(str(dialog.lneXMin.text()))
            xmax = float(str(dialog.lneXMax.text()))
            x_range = (xmin, xmax)
        except ValueError:
            x_range = (None, None)

        try:
            ymin = float(str(dialog.lneYMin.text()))
            ymax = float(str(dialog.lneYMax.text()))
            y_range = (ymin, ymax)
        except ValueError:
            y_range = (None, None)

        try:
            cmin = float(str(dialog.lneCMin.text()))
            cmax = float(str(dialog.lneCMax.text()))
            colorbar_range = (cmin, cmax)
        except ValueError:
            colorbar_range = (None, None)

        legends = LegendDescriptor(visible=dialog.chkShowLegends.isChecked(),
                                   applicable=dialog.groupBox.isHidden())
        for legend_widget in dialog._legend_widgets:
            legends.set_legend_text(handle=legend_widget.handle,
                                    text=legend_widget.get_text(),
                                    visible=legend_widget.is_visible())

        return PlotConfig(title=dialog.lneFigureTitle.text(),
                          x_axis_label=dialog.lneXAxisLabel.text(),
                          y_axis_label=dialog.lneYAxisLabel.text(),
                          legends=legends,
                          errorbars_enabled=dialog.chkShowErrorBars.isChecked(),
                          x_range=x_range,
                          y_range=y_range,
                          colorbar_range=colorbar_range)


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


class LegendDescriptor(object):
    """This is a class that describes the legends on a plot"""
    def __init__(self, visible=False, applicable=True, handles=None):
        self.visible = visible
        self.applicable = applicable
        if handles:
            self.handles = list(handles)
        else:
            self.handles = []
        self._labels = {}

    def all_legends(self):
        """An iterator which yields a dictionary description of legends containing the handle, text and if visible or not"""
        for handle in self.handles:
            yield self.get_legend_descriptor(handle)

    def set_legend_text(self, handle, text, visible=True):
        if handle not in self.handles:
            self.handles.append(handle)
        if not visible:
            text = '_' + text
        self._labels[handle] = text

    def get_legend_descriptor(self, handle):
        if handle in self._labels.keys():
            label = self._labels[handle]  # If a new value has been set for a handle return that
        else:
            label = handle.get_label()   # Else get the value from the plot
        if label.startswith('_'):
            x = {'text': label[1:], 'visible': False, 'handle': handle}
        else:
            x = {'text': label, 'visible': True, 'handle': handle}
        return x

    def get_legend_text(self, handle):
        if handle in self._labels.keys():
            return self._labels[handle]
        return handle.get_label()


class PlotConfig(object):
    def __init__(self, title=None, x_axis_label=None, y_axis_label=None, legends=None, errorbars_enabled=None,
                 x_range=(None, None), y_range=(None, None), colorbar_range=(None, None)):
        self.title = title
        self.xlabel = x_axis_label
        self.ylabel = y_axis_label
        if legends is None:
            self.legend = LegendDescriptor()
        else:
            self.legend = legends
        self.errorbar = errorbars_enabled   # Has 3 values (True : shown, False: Not Shown, None: Not applicable)
        self.x_range = x_range
        self.y_range = y_range
        self.colorbar_range = colorbar_range

    @property
    def title(self):
        if self._title is not None:
            return self._title
        return ""

    @title.setter
    def title(self, value):
        if value is None:
            self._title = None
        else:
            try:
                self._title = str(value)
            except ValueError:
                raise ValueError("Plot title must be a string or castable to string")

    @property
    def xlabel(self):
        if self._xlabel is not None:
            return self._xlabel
        return ""

    @xlabel.setter
    def xlabel(self, value):
        if value is None:
            self._xlabel = None
        else:
            try:
                self._xlabel = str(value)
            except ValueError:
                raise ValueError("Plot xlabel must be a string or castable to string")

    @property
    def ylabel(self):
        if self._ylabel is not None:
            return self._ylabel
        return ""

    @ylabel.setter
    def ylabel(self, value):
        if value is None:
            self._ylabel = None
        else:
            try:
                self._ylabel = str(value)
            except ValueError:
                raise ValueError("Plot ylabel must be a string or castable to string")
