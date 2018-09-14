from matplotlib.container import ErrorbarContainer
from matplotlib.legend import Legend
import warnings
import numpy as np

from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter
from mslice.presenters.quick_options_presenter import quick_options
from mslice.plotting.plot_window.plot_options import CutPlotOptions
from mslice.plotting.plot_window.iplot import IPlot


def get_min(data, absolute_minimum=-np.inf):
    """Determines the minimum value in a set of numpy arrays (ignoring values below absolute_minimum)"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = np.greater(data, absolute_minimum)
    return np.min(np.extract(mask, data))


class CutPlot(IPlot):

    def __init__(self, figure_manager, cut_plotter_presenter, workspace_name):
        self.manager = figure_manager
        self.plot_window = figure_manager.window
        self._canvas = self.plot_window.canvas
        self._cut_plotter_presenter = cut_plotter_presenter
        self._lines_visible = {}
        self._legends_shown = True
        self._legends_visible = []
        self._legend_dict = {}
        self.ws_name = workspace_name
        self._lines = self.line_containers()
        self.setup_connections(self.plot_window)

    def setup_connections(self, plot_window):
        plot_window.menu_intensity.setDisabled(True)
        plot_window.menu_information.setDisabled(True)
        plot_window.action_interactive_cuts.setVisible(False)
        plot_window.action_save_cut.setVisible(False)
        plot_window.action_save_cut.triggered.connect(self.save_icut)
        plot_window.action_flip_axis.setVisible(False)
        plot_window.action_flip_axis.triggered.connect(self.flip_icut)

    def disconnect(self, plot_window):
        plot_window.action_save_cut.triggered.disconnect()
        plot_window.action_flip_axis.triggered.disconnect()

    def window_closing(self):
        icut = self._cut_plotter_presenter.get_icut(self.ws_name)
        if icut is not None:
            icut.window_closing()
            self.manager.button_pressed_connected(False)
            self.manager.picking_connected(False)
        self._canvas.figure.clf()

    def plot_options(self):
        new_config = CutPlotOptionsPresenter(CutPlotOptions(), self).get_new_config()
        if new_config:
            self._canvas.draw()

    def plot_clicked(self, x, y):
        bounds = self.calc_figure_boundaries()
        if bounds['x_label'] < y < bounds['title']:
            if bounds['y_label'] < x:
                if y < bounds['x_range']:
                    quick_options('x_range', self, self.x_log)
                elif x < bounds['y_range']:
                    quick_options('y_range', self, self.y_log)
            self._canvas.draw()

    def object_clicked(self, target):
        if isinstance(target, Legend):
            return
        quick_options(target, self)
        self.update_legend()
        self._canvas.draw()

    def update_legend(self, line_data=None):
        axes = self._canvas.figure.gca()
        labels_to_show = []
        handles_to_show = []
        handles, labels = axes.get_legend_handles_labels()
        if line_data is None:
            i = 0
            for handle, label in zip(handles, labels):
                if self.legend_visible(i):
                    labels_to_show.append(label)
                    handles_to_show.append(handle)
                i += 1
        else:
            containers = axes.containers
            for i in range(len(containers)):
                if line_data[i]['legend']:
                    handles_to_show.append(handles[i])
                    labels_to_show.append(line_data[i]['label'])
                self._legends_visible[i] = line_data[i]['legend']
        axes.legend(handles_to_show, labels_to_show, fontsize='medium').draggable()  # add new legends

    def change_axis_scale(self, xy_config):
        current_axis = self._canvas.figure.gca()
        if xy_config['x_log']:
            xdata = [ll.get_xdata() for ll in current_axis.get_lines()]
            xmin = get_min(xdata, absolute_minimum=0.)
            current_axis.set_xscale('symlog', linthreshx=pow(10, np.floor(np.log10(xmin))))
            if xmin > 0:
                xy_config['x_range'] = (xmin, xy_config['x_range'][1])
        else:
            current_axis.set_xscale('linear')
        if xy_config['y_log']:
            ydata = [ll.get_ydata() for ll in current_axis.get_lines()]
            ymin = get_min(ydata, absolute_minimum=0.)
            current_axis.set_yscale('symlog', linthreshy=pow(10, np.floor(np.log10(ymin))))
            if ymin > 0:
                xy_config['y_range'] = (ymin, xy_config['y_range'][1])
        else:
            current_axis.set_yscale('linear')
        self.x_range = xy_config['x_range']
        self.y_range = xy_config['y_range']

    def get_line_options(self, line):
        index = self._get_line_index(line)
        return self.get_line_options_by_index(index)

    def set_line_options(self, line, line_options):
        index = self._get_line_index(line)
        self.set_line_options_by_index(index, line_options)

    def get_all_line_options(self):
        all_line_options = []
        for i in range(len(self._canvas.figure.gca().containers)):
            line_options = self.get_line_options_by_index(i)
            all_line_options.append(line_options)
        return all_line_options

    def set_all_line_options(self, line_data):
        containers = self._canvas.figure.gca().containers
        for i in range(len(containers)):
            self.set_line_options_by_index(i, line_data[i])
        self.update_legend(line_data)

    def _single_line_has_error_bars(self, line_index):
        current_axis = self._canvas.figure.gca()
        # If all the error bars have alpha = 0 they are all transparent (hidden)
        containers = [x for x in current_axis.containers if isinstance(x, ErrorbarContainer)]
        line_components = [x.get_children() for x in containers]
        # drop the first element of each container because it is the the actual line
        errorbar = [x[1:] for x in line_components][line_index]
        alpha = [x.get_alpha() for x in errorbar]
        # replace None with 1(None indicates default which is 1)
        alpha = [x if x is not None else 1 for x in alpha]
        return sum(alpha) != 0

    def get_line_options_by_index(self, line_index):
        line_options = {}
        container = self._canvas.figure.gca().containers[line_index]
        line = container.get_children()[0]
        line_options['label'] = container.get_label()
        line_options['legend'] = self.legend_visible(line_index)
        line_options['shown'] = self.get_line_visible(line_index)
        line_options['color'] = line.get_color()
        line_options['style'] = line.get_linestyle()
        line_options['width'] = str(int(line.get_linewidth()))
        line_options['marker'] = line.get_marker()
        line_options['error_bar'] = self._single_line_has_error_bars(line_index)
        return line_options

    def set_line_options_by_index(self, line_index, line_options):
        container = self._canvas.figure.gca().containers[line_index]
        container.set_label(line_options['label'])
        main_line = container.get_children()[0]
        main_line.set_linestyle(line_options['style'])
        main_line.set_marker(line_options['marker'])

        self._legends_visible[line_index] = bool(line_options['legend'])

        self.toggle_errorbar(line_index, line_options)

        for child in container.get_children():
            child.set_color(line_options['color'])
            child.set_linewidth(line_options['width'])
            child.set_visible(line_options['shown'])

        self._lines_visible[line_index] = line_options['shown']

    def toggle_errorbar(self, line_index, line_options):
        container = self._canvas.figure.gca().containers[line_index]
        error_bar_elements = container.get_children()[1:]

        if not line_options['error_bar'] and self.get_line_visible(line_index):
            for element in error_bar_elements:
                element.set_alpha(0)
        else:
            for element in error_bar_elements:
                element.set_alpha(1)

    def is_icut(self, is_icut):
        self.manager.button_pressed_connected(not is_icut)
        self.manager.picking_connected(not is_icut)

        self.plot_window.action_save_cut.setVisible(is_icut)
        self.plot_window.action_plot_options.setVisible(not is_icut)
        self.plot_window.keep_make_current_seperator.setVisible(not is_icut)
        self.plot_window.action_keep.setVisible(not is_icut)
        self.plot_window.action_make_current.setVisible(not is_icut)
        self.plot_window.action_flip_axis.setVisible(is_icut)

        self.plot_window.show()

    def save_icut(self):
        icut = self._cut_plotter_presenter.get_icut(self.ws_name)
        return icut.save_cut()

    def flip_icut(self):
        icut = self._cut_plotter_presenter.get_icut(self.ws_name)
        icut.flip_axis()

    def _get_line_index(self, line):
        """
        Checks if line index is cached, and if not finds the index by iterating over the axes' containers.
        :param line: Line to find the index of
        :return: Index of line
        """
        try:
            container = self._lines[line]
        except KeyError:
            self._lines = self.line_containers()
            container = self._lines[line]
        i = 0
        for c in self._canvas.figure.gca().containers:
            if container == c:
                return i
            i += 1

    def calc_figure_boundaries(self):
        fig_x, fig_y = self._canvas.figure.get_size_inches() * self._canvas.figure.dpi
        bounds = {}
        bounds['y_label'] = fig_x * 0.07
        bounds['y_range'] = fig_x * 0.12
        bounds['title'] = fig_y * 0.9
        bounds['x_range'] = fig_y * 0.09
        bounds['x_label'] = fig_y * 0.05
        return bounds

    def xy_config(self):
        return {'x_log': self.x_log, 'y_log': self.y_log, 'x_range': self.x_range, 'y_range': self.y_range}

    def legend_visible(self, index):
        try:
            v = self._legends_visible[index]
        except IndexError:
            v = True
            self._legends_visible.append(True)
        return v

    def line_containers(self):
        """build dictionary of lines and their containers"""
        line_containers = {}
        containers = self._canvas.figure.gca().containers
        for container in containers:
            line = container.get_children()[0]
            line_containers[line] = container
        return line_containers

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
    def show_legends(self):
        return self._legends_shown

    @show_legends.setter
    def show_legends(self, value):
        self._legends_shown = value

    @property
    def title(self):
        return self.manager.title

    @title.setter
    def title(self, value):
        self.manager.title = value

    @property
    def x_label(self):
        return self.manager.x_label

    @x_label.setter
    def x_label(self, value):
        self.manager.x_label = value

    @property
    def y_label(self):
        return self.manager.y_label

    @y_label.setter
    def y_label(self, value):
        self.manager.y_label = value

    @property
    def x_range(self):
        return self.manager.x_range

    @x_range.setter
    def x_range(self, value):
        self.manager.x_range = value

    @property
    def y_range(self):
        return self.manager.y_range

    @y_range.setter
    def y_range(self, value):
        self.manager.y_range = value

    @property
    def x_grid(self):
        return self.manager.x_grid

    @x_grid.setter
    def x_grid(self, value):
        self.manager.x_grid = value

    @property
    def y_grid(self):
        return self.manager.y_grid

    @y_grid.setter
    def y_grid(self, value):
        self.manager.y_grid = value