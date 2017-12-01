from itertools import chain

from matplotlib.container import ErrorbarContainer
import numpy as np

from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter
from .plot_options import CutPlotOptions


class CutPlot(object):

    def __init__(self, plot_figure, canvas, cut_plotter):
        self.plot_figure = plot_figure
        self.canvas = canvas
        self.cut_plotter = cut_plotter
        self.lines_visible = {}
        self.legends_shown = True
        self.legends_visible = []
        self.legend_dict = {}
        plot_figure.menuIntensity.setDisabled(True)
        plot_figure.menuInformation.setDisabled(True)

    def plot_options(self):
        new_config = CutPlotOptionsPresenter(CutPlotOptions(), self).get_new_config()
        if new_config:
            self.canvas.draw()

    @staticmethod
    def get_min(data, absolute_minimum=-np.inf):
        """Determines the minimum of a set of numpy arrays"""
        data = data if isinstance(data, list) else [data]
        running_min = []
        for values in data:
            try:
                running_min.append(np.min(values[np.isfinite(values) * (values > absolute_minimum)]))
            except ValueError:  # If data is empty or not array of numbers
                pass
        return np.min(running_min) if running_min else absolute_minimum

    def change_cut_plot(self, xy_config):
        current_axis = self.canvas.figure.gca()
        if xy_config['x_log']:
            xdata = [ll.get_xdata() for ll in current_axis.get_lines()]
            xmin = self.get_min(xdata, absolute_minimum=0.)
            current_axis.set_xscale('symlog', linthreshx=pow(10, np.floor(np.log10(xmin))))
            if xmin > 0:
                xy_config['x_range'] = (xmin, xy_config['x_range'][1])
        else:
            current_axis.set_xscale('linear')
        if xy_config['y_log']:
            ydata = [ll.get_ydata() for ll in current_axis.get_lines()]
            ymin = self.get_min(ydata, absolute_minimum=0.)
            current_axis.set_yscale('symlog', linthreshy=pow(10, np.floor(np.log10(ymin))))
            if ymin > 0:
                xy_config['y_range'] = (ymin, xy_config['y_range'][1])
        else:
            current_axis.set_yscale('linear')
        self.x_range = xy_config['x_range']
        self.y_range = xy_config['y_range']

    def _has_errorbars(self):
        """True current axes has visible errorbars,
         False if current axes has hidden errorbars"""
        current_axis = self.canvas.figure.gca()
        # If all the error bars have alpha= 0 they are all transparent (hidden)
        containers = [x for x in current_axis.containers if isinstance(x, ErrorbarContainer)]
        line_components = [x.get_children() for x in containers]
        # drop the first element of each container because it is the the actual line
        errorbars = [x[1:] for x in line_components]
        errorbars = chain(*errorbars)
        alpha = [x.get_alpha() for x in errorbars]
        # replace None with 1(None indicates default which is 1)
        alpha = [x if x is not None else 1 for x in alpha]
        if sum(alpha) == 0:
            has_errorbars = False
        else:
            has_errorbars = True
        return has_errorbars

    def _set_errorbars_shown_state(self, state):
        """Show errrorbar if state = 1, hide if state = 0"""
        current_axis = self.canvas.figure.gca()
        if state:
            alpha = 1
        else:
            alpha = 0.
        for i in range(len(current_axis.containers)):
            if isinstance(current_axis.containers[i], ErrorbarContainer):
                elements = current_axis.containers[i].get_children()
                if self.get_line_visible(i):
                    elements[1].set_alpha(alpha)  # elements[0] is the actual line, elements[1] is error bars

    def _toggle_errorbars(self):
        state = self._has_errorbars()
        if state is None:  # No errorbars in this plot
            return
        self._set_errorbars_shown_state(not state)

    def get_legends(self):
        current_axis = self.canvas.figure.gca()
        legends = []
        labels = current_axis.get_legend_handles_labels()[1]
        for i in range(len(labels)):
            try:
                v = self.legends_visible[i]
            except IndexError:
                v = True
                self.legends_visible.append(True)
            legends.append({'label': labels[i], 'visible': v})
        return legends

    def set_legends(self, legends):
        current_axes = self.canvas.figure.gca()
        if current_axes.legend_:
            current_axes.legend_.remove()  # remove old legends
        if legends is None or not self.legends_shown:
            return
        labels = []
        handles_to_show = []
        handles = current_axes.get_legend_handles_labels()[0]
        for i in range(len(legends)):
            container = current_axes.containers[i]
            container.set_label(legends[i]['label'])
            if legends[i]['visible']:
                handles_to_show.append(handles[i])
                labels.append(legends[i]['label'])
            self.legends_visible[i] = legends[i]['visible']
        x = current_axes.legend(handles_to_show, labels)  # add new legends
        x.draggable()

    def toggle_legend(self):
        self.legends_shown = not self.legends_shown
        self.set_legends(self.get_legends())
        self.canvas.draw()

    def get_line_data(self):
        legends = self.get_legends()
        all_line_options = []
        i = 0
        for line_group in self.canvas.figure.gca().containers:
            line_options = {}
            line = line_group.get_children()[0]
            line_options['shown'] = self.get_line_visible(i)
            line_options['color'] = line.get_color()
            line_options['style'] = line.get_linestyle()
            line_options['width'] = str(int(line.get_linewidth()))
            line_options['marker'] = line.get_marker()
            all_line_options.append(line_options)
            i += 1
        return list(zip(legends, all_line_options))

    def set_line_data(self, line_data):
        legends = []
        i = 0
        for line in line_data:
            legend, line_options = line
            legends.append(legend)
            line_model = self.canvas.figure.gca().containers[i]
            self.set_line_visible(i, line_options['shown'])
            for child in line_model.get_children():
                child.set_color(line_options['color'])
                child.set_linewidth(line_options['width'])
            main_line = line_model.get_children()[0]
            main_line.set_linestyle(line_options['style'])
            main_line.set_marker(line_options['marker'])
            i += 1
        self.set_legends(legends)

    def set_line_visible(self, line_index, visible):
        self.lines_visible[line_index] = visible
        for child in self.canvas.figure.gca().containers[line_index].get_children():
            child.set_visible(visible)

    def get_line_visible(self, line_index):
        try:
            ret = self.lines_visible[line_index]
            return ret
        except KeyError:
            self.lines_visible[line_index] = True
            return True

    @property
    def x_log(self):
        return 'log' in self.canvas.figure.gca().get_xscale()

    @property
    def y_log(self):
        return 'log' in self.canvas.figure.gca().get_yscale()

    @property
    def error_bars(self):
        return self._has_errorbars()

    @error_bars.setter
    def error_bars(self, value):
        self._set_errorbars_shown_state(value)

    @property
    def show_legends(self):
        return self.legends_shown

    @show_legends.setter
    def show_legends(self, value):
        self.legends_shown = value

    @property
    def title(self):
        return self.plot_figure.title

    @title.setter
    def title(self, value):
        self.plot_figure.title = value

    @property
    def x_label(self):
        return self.plot_figure.x_label

    @x_label.setter
    def x_label(self, value):
        self.plot_figure.x_label = value

    @property
    def y_label(self):
        return self.plot_figure.y_label

    @y_label.setter
    def y_label(self, value):
        self.plot_figure.y_label = value

    @property
    def x_range(self):
        return self.plot_figure.x_range

    @x_range.setter
    def x_range(self, value):
        self.plot_figure.x_range = value

    @property
    def y_range(self):
        return self.plot_figure.y_range

    @y_range.setter
    def y_range(self, value):
        self.plot_figure.y_range = value
