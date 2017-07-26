from mslice.plotting.plot_window.plot_options import PlotOptionsDialog

class PlotOptionsPresenter(object):
    def __init__(self, current_config, PlotOptionsDialog):

        # propagate dialog with existing data
        self._plot_options_dialog = PlotOptionsDialog
        self._plot_options_dialog.title = current_config.title
        self._plot_options_dialog.x_label = current_config.xlabel
        self._plot_options_dialog.y_label = current_config.ylabel
        self._plot_options_dialog.x_range = (current_config.x_range[0], current_config.x_range[1])
        self._plot_options_dialog.y_range = (current_config.y_range[0], current_config.y_range[1])
        if None not in current_config.colorbar_range:
            self._plot_options_dialog.colorbar_range = (current_config.colorbar_range[0], current_config.colorbar_range[1])
            self._plot_options_dialog.chkLogarithmic.setChecked(current_config.logarithmic)
            self._plot_options_dialog.chkXLog.hide()
            self._plot_options_dialog.chkYLog.hide()
        else:
            self._plot_options_dialog.groupBox_4.hide()
            self._plot_options_dialog.chkXLog.setChecked(current_config.xlog)
            self._plot_options_dialog.chkYLog.setChecked(current_config.ylog)

        self._plot_options_dialog.chkShowLegends.setChecked(current_config.legend.visible)
        if current_config.errorbar is None:
            self._plot_options_dialog.chkShowErrorBars.hide()
        else:
            self._plot_options_dialog.chkShowErrorBars.setChecked(current_config.errorbar)
        if not current_config.legend.applicable:
            self._plot_options_dialog.groupBox.hide()
        else:
            self._plot_options_dialog.chkShowLegends.setChecked(current_config.legend.visible)
            for legend in current_config.legend.all_legends():
                legend_widget = self._plot_options_dialog.add_legend(legend['text'], legend['handle'], legend['visible'])

    def get_new_config(self):
        dialog_accepted = self._plot_options_dialog.exec_()
        if not dialog_accepted:
            return None
        legends = LegendDescriptor( visible=self._plot_options_dialog.chkShowLegends.isChecked(),
                                    applicable=self._plot_options_dialog.groupBox.isHidden())
        for legend_widget in self._plot_options_dialog._legend_widgets:
            legends.set_legend_text(handle=legend_widget.handle,
                                    text=legend_widget.get_text(),
                                    visible=legend_widget.is_visible())

        return PlotConfig(title=self._plot_options_dialog.title,
                          xlabel=self._plot_options_dialog.x_label,
                          ylabel=self._plot_options_dialog.y_label,
                          legend=legends,
                          errorbar=self._plot_options_dialog.chkShowErrorBars.isChecked(),
                          x_range=self._plot_options_dialog.x_range, xlog=self._plot_options_dialog.chkXLog.isChecked(),
                          y_range=self._plot_options_dialog.y_range, ylog=self._plot_options_dialog.chkYLog.isChecked(),
                          colorbar_range=self._plot_options_dialog.colorbar_range,
                          logarithmic=self._plot_options_dialog.chkLogarithmic.isChecked())

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
    def __init__(self, **kwargs):
        # Define default values for all options
        self.title = None
        self.xlabel = None
        self.ylabel = None
        self.xlog = False
        self.ylog = False
        self.legend = LegendDescriptor()
        self.errorbar = None
        self.x_range = None
        self.y_range = None
        self.colorbar_range = None
        self.logarithmic = False
        # Populates fields from keyword arguments
        for (argname, value) in kwargs.items():
            if value is not None:
                setattr(self, argname, value)

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