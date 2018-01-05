from __future__ import (absolute_import, division, print_function)
from functools import partial


class PlotOptionsPresenter(object):
    def __init__(self, plot_options_dialog, plot_handler):
        self._model = plot_handler
        self._view = plot_options_dialog
        self._modified_values = {}
        self._xy_config = {'x_range': self._model.x_range, 'y_range': self._model.y_range, 'modified': False}

        self.set_properties()  # propagate dialog with existing data

        self._view.titleEdited.connect(partial(self._value_modified, 'title'))
        self._view.xLabelEdited.connect(partial(self._value_modified, 'x_label'))
        self._view.yLabelEdited.connect(partial(self._value_modified, 'y_label'))
        self._view.xRangeEdited.connect(partial(self._xy_config_modified, 'x_range'))
        self._view.yRangeEdited.connect(partial(self._xy_config_modified, 'y_range'))
        self._view.xGridEdited.connect(partial(self._value_modified, 'x_grid'))
        self._view.yGridEdited.connect(partial(self._value_modified, 'y_grid'))

    def _value_modified(self, value_name):
        self._modified_values[value_name] = getattr(self._view, value_name)

    def _xy_config_modified(self, key):
        getattr(self, '_xy_config')[key] = getattr(self._view, key)
        self._xy_config['modified'] = True


class SlicePlotOptionsPresenter(PlotOptionsPresenter):

    def __init__(self, plot_options_dialog, slice_handler):
        super(SlicePlotOptionsPresenter, self).__init__(plot_options_dialog, slice_handler)

        self._color_config = {'c_range': self._model.colorbar_range, 'log': self._model.colorbar_log, 'modified': False}

        self._view.cLogEdited.connect(self._set_colorbar_log)
        self._view.cRangeEdited.connect(self._set_c_range)

    def set_properties(self):
        properties = ['title', 'x_label', 'y_label', 'x_range', 'y_range', 'x_grid', 'y_grid',
                      'colorbar_range', 'colorbar_log']
        for p in properties:
            setattr(self._view, p, getattr(self._model, p))

    def get_new_config(self):
        dialog_accepted = self._view.exec_()
        if not dialog_accepted:
            return None
        if self._color_config['modified']:
            self._model.change_axis_scale(self._color_config['c_range'], self._color_config['log'])
        for key, value in list(self._modified_values.items()):
            setattr(self._model, key, value)
        self._model.x_range = self._xy_config['x_range']
        self._model.y_range = self._xy_config['y_range']
        return True

    def _set_c_range(self):
        self._color_config['c_range'] = self._view.colorbar_range
        self._color_config['modified'] = True

    def _set_colorbar_log(self):
        self._color_config['log'] = self._view.colorbar_log
        self._color_config['modified'] = True


class CutPlotOptionsPresenter(PlotOptionsPresenter):

    def __init__(self, plot_options_dialog, cut_handler):
        super(CutPlotOptionsPresenter, self).__init__(plot_options_dialog, cut_handler)
        self._xy_config.update({'x_log': self._model.x_log, 'y_log': self._model.y_log})

        self._view.showLegendsEdited.connect(partial(self._value_modified, 'show_legends'))
        self._view.xLogEdited.connect(partial(self._xy_config_modified, 'x_log'))
        self._view.yLogEdited.connect(partial(self._xy_config_modified, 'y_log'))

        line_data = self._model.get_all_line_data()
        self._view.set_line_data(line_data)

    def set_properties(self):
        properties = ['title', 'x_label', 'y_label', 'x_range', 'y_range', 'x_log', 'y_log',
                      'x_grid', 'y_grid', 'error_bars', 'show_legends']
        for p in properties:
            setattr(self._view, p, getattr(self._model, p))

    def get_new_config(self):
        dialog_accepted = self._view.exec_()
        if not dialog_accepted:
            return None
        if self._xy_config['modified']:
            self._model.change_axis_scale(self._xy_config)
        for key, value in list(self._modified_values.items()):
            setattr(self._model, key, value)
        line_data = self._view.get_line_data()
        self._model.set_all_line_data(line_data)
        self._model.error_bars = self._view.error_bars
        return True
