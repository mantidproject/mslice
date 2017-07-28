import matplotlib.colors as colors
from functools import partial


class PlotOptionsPresenter(object):
    def __init__(self, plot_options_dialog, plot_figure_manager):

        self._model = plot_figure_manager
        self._plot_window_canvas = plot_figure_manager.canvas.figure.gca()
        self._view = plot_options_dialog

        self._modified_values = {}
        self._xy_log = {'x_log': self.x_log,           'y_log': self.y_log,
                        'x_range': self.x_range,       'y_range': self.y_range,
                        'error_bars': self.error_bars, 'modified': False }
        self._color_log = {'c_range': self.colorbar_range, 'log': self.logarithmic, 'modified': False}

        # propagate dialog with existing data
        properties = ['title', 'x_label', 'y_label', 'x_range', 'y_range']

        if None not in self.colorbar_range:
            properties.append('colorbar_range')
            self._view.set_log('c', self.logarithmic)
        else:
            self._view.set_log('x', self.x_log)
            self._view.set_log('y', self.y_log)
            self._view.set_show_error_bars(self.error_bars)

            legends = self._model.get_legends()
            self._view.set_legends(legends)

        for p in properties:
            setattr(self._view, p, getattr(self, p))

        self._view.titleEdited.connect(partial(self._value_modified, 'title'))
        self._view.xLabelEdited.connect(partial(self._value_modified, 'x_label'))
        self._view.yLabelEdited.connect(partial(self._value_modified, 'y_label'))
        self._view.errorBarsEdited.connect(partial(self._value_modified, 'error_bars'))
        self._view.xLogEdited.connect(self._set_x_log)
        self._view.yLogEdited.connect(self._set_y_log)
        self._view.cLogEdited.connect(self._set_c_log)
        self._view.xRangeEdited.connect(self._set_x_range)
        self._view.yRangeEdited.connect(self._set_y_range)
        self._view.cRangeEdited.connect(self._set_c_range)


    def _value_modified(self, value_name):
        self._modified_values[value_name] = getattr(self._view, value_name)

    def _set_x_log(self): #probs conciser way of writing these 4
        self._xy_log['x_log'] = self._view.x_log
        self._xy_log['modified'] = True

    def _set_y_log(self):
        self._xy_log['y_log'] = self._view.y_log
        self._xy_log['modified'] = True

    def _set_x_range(self):
        self._xy_log['x_range'] = self._view.x_range
        self._xy_log['modified'] = True

    def _set_y_range(self):
        self._xy_log['y_range'] = self._view.y_range
        self._xy_log['modified'] = True

    def _set_c_range(self):
        self._color_log['c_range'] = self._view.colorbar_range
        self._color_log['modified'] = True

    def _set_c_log(self):
        self._color_log['log'] = self._view.c_log
        self._color_log['modified'] = True

    def get_new_config(self):
        dialog_accepted = self._view.exec_()
        if not dialog_accepted:
            return None
        if self._view.color_plot:
            if self._color_log['modified']:
                self._model.change_colorplot(self._color_log['c_range'], self._color_log['log'])
                for key, value in self._modified_values.items():
                    setattr(self, key, value)
            self._model.set_x_range(self._xy_log['x_range'])
            self._model.set_y_range(self._xy_log['y_range'])

        else:
            if self._xy_log['modified']:
                self._model.change_lineplot(self._xy_log)
            if self._xy_log['error_bars'] is not None:
                self._model._set_errorbars_shown_state(self._xy_log['error_bars'])
            for key, value in self._modified_values.items():
                setattr(self, key, value)
            legends = self._view.get_legends()
            self._model.set_legends(legends)
        return True

    @property
    def title(self):
        return self._plot_window_canvas.get_title()

    @title.setter
    def title(self, value):
        self._plot_window_canvas.set_title(value)

    @property
    def x_label(self):
        return self._plot_window_canvas.get_xlabel()

    @x_label.setter
    def x_label(self, value):
        self._plot_window_canvas.set_xlabel(value)

    @property
    def y_label(self):
        return self._plot_window_canvas.get_ylabel()

    @y_label.setter
    def y_label(self, value):
        self._plot_window_canvas.set_ylabel(value)

    @property
    def x_range(self):
        return self._plot_window_canvas.get_xlim()

    @property
    def y_range(self):
        return self._plot_window_canvas.get_ylim()

    @property
    def colorbar_range(self):
        if self._plot_window_canvas.get_images():
            return self._plot_window_canvas.get_images()[0].get_clim()
        else:
            return None, None

    @colorbar_range.setter
    def colorbar_range(self, value):
        self._color_log['c_range'] = value
        self._color_log['modified'] = True

    @property
    def logarithmic(self):
        if self._plot_window_canvas.get_images():
            mappable = self._plot_window_canvas.get_images()[0]
            norm = mappable.norm
            return isinstance(norm, colors.LogNorm)
        else:
            return False

    @logarithmic.setter
    def logarithmic(self, value):
        self._color_log['log'] = value
        self._color_log['modified'] = True

    @property
    def x_log(self):
        if not self._plot_window_canvas.get_images():
            return 'log' in self._plot_window_canvas.get_xscale()
        else:
            return False

    @property
    def y_log(self):
        if not self._plot_window_canvas.get_images():
            return 'log' in self._plot_window_canvas.get_yscale()
        else:
            return False

    @property
    def error_bars(self):
        return self._model._has_errorbars()  # maybe some of these props should be in plot_figure

    @error_bars.setter
    def error_bars(self, value):
        self._xy_log['error_bars'] = value
        self._xy_log['modified'] = True


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
        """An iterator which yields a dictionary description of legends containing the handle,
        text and if visible or not"""
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
