from itertools import chain

from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
import numpy as np

from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter
from mslice.presenters.quick_options_presenter import quick_options
from .plot_options import CutPlotOptions


class CutPlot(object):

    def __init__(self, plot_figure, canvas, cut_plotter):
        self.plot_figure = plot_figure
        self._canvas = canvas
        self._cut_plotter = cut_plotter
        self._lines_visible = {}
        self._legends_shown = True
        self._legends_visible = []
        self._legend_dict = {}
        self._lines = self.line_containers()
        plot_figure.menuIntensity.setDisabled(True)
        plot_figure.menuInformation.setDisabled(True)
        self.update_legend()
        np.seterr(invalid='ignore')

    def plot_options(self):
        new_config = CutPlotOptionsPresenter(CutPlotOptions(), self).get_new_config()
        if new_config:
            self._canvas.draw()

    def object_clicked(self, target):
        leg_pos = self._canvas.figure.gca().get_legend()._loc
        print(type(target))
        if isinstance(target, Line2D):
            target = self.get_line_container(target)
        self._quick_presenter = quick_options(target, self)
        # self.update_legend()
        self._canvas.figure.gca().get_legend()._loc = leg_pos
        self._canvas.draw()

    def get_line_container(self, line):
        try:
            return self._lines[line]
        except KeyError:
            self._lines=self.line_containers()
            return self._lines[line]


    def plot_clicked(self, x, y):
        bounds = self.calc_figure_boundaries()
        if bounds['x_label'] < y < bounds['title']:
            if bounds['y_label'] < x:
                if y < bounds['x_range']:
                    self._quick_presenter = quick_options('x_range', self, self.x_log)
                elif x < bounds['y_range']:
                    self._quick_presenter = quick_options('y_range', self, self.y_log)
            self._canvas.draw()

    def calc_figure_boundaries(self):
        fig_x, fig_y = self._canvas.figure.get_size_inches() * self._canvas.figure.dpi
        bounds = {}
        bounds['y_label'] = fig_x * 0.07
        bounds['y_range'] = fig_x * 0.12
        bounds['title'] = fig_y * 0.9
        bounds['x_range'] = fig_y * 0.09
        bounds['x_label'] = fig_y * 0.05
        return bounds

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

    def xy_config(self):
        return {'x_log': self.x_log, 'y_log': self.y_log, 'x_range': self.x_range, 'y_range': self.y_range}

    def change_axis_scale(self, xy_config):
        current_axis = self._canvas.figure.gca()
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
        current_axis = self._canvas.figure.gca()
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
        current_axis = self._canvas.figure.gca()
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
        current_axis = self._canvas.figure.gca()
        legends = []
        labels = current_axis.get_legend_handles_labels()[1]
        for i in range(len(labels)):
            try:
                v = self._legends_visible[i]
            except IndexError:
                v = True
                self._legends_visible.append(True)
            legends.append({'label': labels[i], 'visible': v})
        return legends

    def set_legends(self, legends):
        current_axes = self._canvas.figure.gca()
        if current_axes.legend_:
            current_axes.legend_.remove()  # remove old legends
        if legends is None or not self._legends_shown:
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
            self._legends_visible[i] = legends[i]['visible']
        x = current_axes.legend(handles_to_show, labels)  # add new legends
        x.draggable()

    def toggle_legend(self):
        self._legends_shown = not self._legends_shown
        self.set_legends(self.get_legends())
        self._canvas.draw()

    def line_containers(self):
        line_containers = {}
        containers = self._canvas.figure.gca().containers
        for index in range(len(containers)):
            container = containers[index]
            line = container.get_children()[0]
            line_containers[line] = container
        return line_containers

    def get_all_line_data(self):
        all_line_options = []
        containers = self._canvas.figure.gca().containers
        for i in range(len(containers)):
            line_options = self.get_line_data(containers[i])
            all_line_options.append(line_options)
        return all_line_options

    def get_line_data(self, container):
        line_options = {}
        line = container.get_children()[0]
        line_options['label'] = container.get_label()
        line_options['shown'] = True
        line_options['color'] = line.get_color()
        line_options['style'] = line.get_linestyle()
        line_options['width'] = str(int(line.get_linewidth()))
        line_options['marker'] = line.get_marker()
        return line_options

    def set_line_data(self, container, line_options):
        container.set_label(line_options['label'])
        main_line = container.get_children()[0]
        main_line.set_linestyle(line_options['style'])
        main_line.set_marker(line_options['marker'])
        for child in container.get_children():
            child.set_color(line_options['color'])
            child.set_linewidth(line_options['width'])
            child.set_visible(line_options['shown'])

    def set_all_line_data(self, list_of_line_options):
        containers = self._canvas.figure.gca().containers
        i = 0
        for line_options in list_of_line_options:
            self.set_line_data(containers[i], line_options)
            i+=1

    def update_legend(self):
        leg = self._canvas.figure.gca().legend(fontsize='medium')
        leg.draggable()


    def set_line_visible(self, line_index, visible):
        self._lines_visible[line_index] = visible
        for child in self._canvas.figure.gca().containers[line_index].get_children():
            child.set_visible(visible)

    def get_line_visible(self, line_index):
        try:
            ret = self._lines_visible[line_index]
            return ret
        except KeyError:
            self._lines_visible[line_index] = True
            return True

    @property
    def x_log(self):
        return 'log' in self._canvas.figure.gca().get_xscale()

    @x_log.setter
    def x_log(self, value):
        config = self.xy_config()
        config['x_log'] = value
        self.change_axis_scale(config)
        self._canvas.draw()

    @property
    def y_log(self):
        return 'log' in self._canvas.figure.gca().get_yscale()

    @y_log.setter
    def y_log(self, value):
        config = self.xy_config()
        config['y_log'] = value
        self.change_axis_scale(config)
        self._canvas.draw()

    @property
    def error_bars(self):
        return self._has_errorbars()

    @error_bars.setter
    def error_bars(self, value):
        self._set_errorbars_shown_state(value)

    @property
    def show_legends(self):
        return self._legends_shown

    @show_legends.setter
    def show_legends(self, value):
        self._legends_shown = value

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

