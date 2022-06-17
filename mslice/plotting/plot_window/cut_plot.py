from distutils.version import LooseVersion
from functools import partial

from matplotlib import __version__ as mpl_version
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text

import numpy as np

from mslice.models.colors import to_hex, name_to_color
from mslice.presenters.plot_options_presenter import CutPlotOptionsPresenter
from mslice.presenters.quick_options_presenter import quick_options, check_latex
from mslice.plotting.plot_window.plot_options import CutPlotOptions
from mslice.plotting.plot_window.iplot import IPlot
from mslice.plotting.plot_window.overplot_interface import toggle_overplot_line, \
    cif_file_powder_line
from mslice.scripting import generate_script
from mslice.util.compat import legend_set_draggable


def get_min(data, absolute_minimum=-np.inf):
    """Determines the minimum value in a set of numpy arrays (ignoring values below absolute_minimum)"""
    masked_data = [np.extract(np.greater(row, absolute_minimum), row) for row in data]
    return np.min([np.min(row) for row in masked_data])


class CutPlot(IPlot):

    def __init__(self, figure_manager, cut_plotter_presenter, workspace_name):
        self.manager = figure_manager
        self.plot_window = figure_manager.window
        self._canvas = self.plot_window.canvas
        self._cut_plotter_presenter = cut_plotter_presenter
        self._plot_options_view = None
        self._lines_visible = {}
        self._legends_shown = True
        self._legends_visible = []
        self._legend_dict = {}
        self.ws_name = workspace_name
        self.ws_list = [workspace_name]
        self._lines = self.line_containers()
        self.setup_connections(self.plot_window)
        self.default_options = None
        self._waterfall_cache = {}
        self._is_icut = False
        self._powder_lines = {}
        self._datum_dirty = True
        self._datum_cache = 0

    def save_default_options(self):
        self.default_options = {
            'legend': True,
            'x_log': False,
            'y_log': False,
            'title': self.ws_name,
            'x_label': r"$|Q|$ ($\mathrm{\AA}^{-1}$)",
            'x_grid': False,
            'x_range': (None, None),
            'y_label': 'Energy Transfer (meV)',
            'y_grid': False,
            'y_range': (None, None),
            'waterfall': False,
        }

    def setup_connections(self, plot_window):
        plot_window.redraw.connect(self._canvas.draw)
        plot_window.menu_information.setDisabled(False)
        plot_window.menu_recoil_lines.setDisabled(True)
        plot_window.menu_intensity.setDisabled(True)
        plot_window.action_toggle_legends.setVisible(True)
        plot_window.action_keep.setVisible(True)
        plot_window.action_make_current.setVisible(True)
        plot_window.action_save_image.setVisible(True)
        plot_window.action_plot_options.setVisible(True)
        plot_window.action_interactive_cuts.setVisible(False)
        plot_window.action_save_cut.setVisible(False)
        plot_window.action_save_cut.triggered.connect(self.save_icut)
        plot_window.action_flip_axis.setVisible(False)
        plot_window.action_flip_axis.triggered.connect(self.flip_icut)
        plot_window.action_gen_script.triggered.connect(self.generate_script)
        plot_window.action_gen_script_clipboard.triggered.connect(lambda: self.generate_script(clipboard=True))
        plot_window.action_waterfall.triggered.connect(self.toggle_waterfall)
        plot_window.waterfall_x_edt.editingFinished.connect(self.toggle_waterfall)
        plot_window.waterfall_y_edt.editingFinished.connect(self.toggle_waterfall)
        plot_window.waterfall_x_edt.editingFinished.connect(plot_window.lose_waterfall_x_edt_focus)
        plot_window.waterfall_y_edt.editingFinished.connect(plot_window.lose_waterfall_y_edt_focus)
        plot_window.action_aluminium.triggered.connect(
            partial(toggle_overplot_line, self, self._cut_plotter_presenter, 'Aluminium', False))
        plot_window.action_copper.triggered.connect(
            partial(toggle_overplot_line, self, self._cut_plotter_presenter, 'Copper', False))
        plot_window.action_niobium.triggered.connect(
            partial(toggle_overplot_line, self, self._cut_plotter_presenter, 'Niobium', False))
        plot_window.action_tantalum.triggered.connect(
            partial(toggle_overplot_line, self, self._cut_plotter_presenter, 'Tantalum', False))
        plot_window.action_cif_file.triggered.connect(partial(cif_file_powder_line, self,
                                                              self._cut_plotter_presenter))

    def disconnect(self, plot_window):
        plot_window.action_save_cut.triggered.disconnect()
        plot_window.action_flip_axis.triggered.disconnect()
        plot_window.action_gen_script.triggered.disconnect()
        plot_window.action_aluminium.triggered.disconnect()
        plot_window.action_copper.triggered.disconnect()
        plot_window.action_niobium.triggered.disconnect()
        plot_window.action_tantalum.triggered.disconnect()
        plot_window.action_cif_file.triggered.disconnect()

    def window_closing(self):
        icut = self._cut_plotter_presenter.get_icut()
        if icut is not None and self._is_icut:
            icut.window_closing()
            self.manager.button_pressed_connected(False)
            self.manager.picking_connected(False)
        self.plot_window.close()

    def plot_options(self):
        self._plot_options_view = CutPlotOptions(self.plot_window, redraw_signal=self.plot_window.redraw)
        return CutPlotOptionsPresenter(self._plot_options_view, self)

    def plot_clicked(self, x, y):
        bounds = self.calc_figure_boundaries()
        if bounds['x_label'] < y < bounds['title']:
            if bounds['y_label'] < x:
                if y < bounds['x_range']:
                    quick_options('x_range', self, self.x_log, redraw_signal=self.plot_window.redraw)
                elif x < bounds['y_range']:
                    quick_options('y_range', self, self.y_log, redraw_signal=self.plot_window.redraw)
            self._canvas.draw()

    def object_clicked(self, target):
        if isinstance(target, Legend):
            return
        elif isinstance(target, Text):
            quick_options(target, self, redraw_signal=self.plot_window.redraw)
        else:
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
        legend = axes.legend(handles_to_show, labels_to_show, fontsize='medium')  # add new legends
        legend_set_draggable(legend, True)

    def change_axis_scale(self, xy_config):
        current_axis = self._canvas.figure.gca()
        orig_y_scale_log = self.y_log
        if xy_config['x_log']:
            xmin = xy_config['x_range'][0]
            xdata = [ll.get_xdata() for ll in current_axis.get_lines()]
            self.x_axis_min = get_min(xdata, absolute_minimum=0.)
            linthresh_val = pow(10, np.floor(np.log10(self.x_axis_min)))

            kwargs = {'linthreshx': linthresh_val} if LooseVersion(mpl_version) < LooseVersion('3.3') \
                else {'linthresh': linthresh_val}
            current_axis.set_xscale('symlog', **kwargs)

            if xmin > 0:
                xy_config['x_range'] = (xmin, xy_config['x_range'][1])
        else:
            current_axis.set_xscale('linear')
        if xy_config['y_log']:
            ymin = xy_config['y_range'][0]
            ydata = [ll.get_ydata() for ll in current_axis.get_lines()]
            self.y_axis_min = get_min(ydata, absolute_minimum=0.)
            linthresh_val = pow(10, np.floor(np.log10(self.y_axis_min)))

            kwargs = {'linthreshy': linthresh_val} if LooseVersion(mpl_version) < LooseVersion('3.3') \
                else {'linthresh': linthresh_val}
            current_axis.set_yscale('symlog', **kwargs)

            if ymin > 0:
                xy_config['y_range'] = (ymin, xy_config['y_range'][1])
        else:
            current_axis.set_yscale('linear')
        self.x_range = xy_config['x_range']
        self.y_range = xy_config['y_range']

        if xy_config['y_log'] or (xy_config['y_log'] != orig_y_scale_log):
            self.update_bragg_peaks(refresh=True)

    def get_line_options(self, line):
        index = self._get_line_index(line)
        if index >= 0:
            return self.get_line_options_by_index(index)
        else:
            line_options = {
                'label': line.get_label(),
                'legend': None,
                'shown': None,
                'color': to_hex(line.get_color()),
                'style': line.get_linestyle(),
                'width': str(int(line.get_linewidth())),
                'marker': line.get_marker(),
                'error_bar': None
            }
            return line_options

    def set_line_options(self, line, line_options):
        index = self._get_line_index(line)
        if index >= 0:
            self.set_line_options_by_index(index, line_options)
        else:
            line.set_label(line_options['label'])
            line.set_linestyle(line_options['style'])
            line.set_marker(line_options['marker'])
            line.set_color(name_to_color(line_options['color']))
            line.set_linewidth(line_options['width'])

    def get_all_line_options(self):
        all_line_options = []
        for i in range(len(self._canvas.figure.gca().containers)):
            line_options = self.get_line_options_by_index(i)
            all_line_options.append(line_options)
        return all_line_options

    def set_all_line_options(self, line_data, update_legend):
        containers = self._canvas.figure.gca().containers
        for i in range(len(containers)):
            self.set_line_options_by_index(i, line_data[i])
        if update_legend:
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
        container = self._canvas.figure.gca().containers[line_index]
        line = container.get_children()[0]
        line_options = {
            'label': container.get_label(),
            'legend': self.legend_visible(line_index),
            'shown': self.get_line_visible(line_index),
            'color': to_hex(line.get_color()),
            'style': line.get_linestyle(),
            'width': str(line.get_linewidth()),
            'marker': line.get_marker(),
            'error_bar': self._single_line_has_error_bars(line_index)
        }
        return line_options

    def set_line_options_by_index(self, line_index, line_options):
        container = self._canvas.figure.gca().containers[line_index]
        container.set_label(line_options['label'])
        main_line = container.get_children()[0]
        main_line.set_linestyle(line_options['style'])
        main_line.set_marker(line_options['marker'])

        try:
            self._legends_visible[line_index] = bool(line_options['legend'])
        except IndexError:
            self._legends_visible.append(bool(line_options['legend']))

        self.toggle_errorbar(line_index, line_options)

        for child in container.get_children():
            child.set_color(name_to_color(line_options['color']))
            child.set_linewidth(line_options['width'])
            child.set_visible(line_options['shown'])

        self._lines_visible[line_index] = line_options['shown']

    def remove_line_by_index(self, line_index):
        containers = self._canvas.figure.gca().containers
        if line_index < len(containers):
            container = containers[line_index]
            container[0].remove()
            for line in container[1] + container[2]:
                line.remove()
            containers.remove(container)

        self._datum_dirty = True
        self.update_bragg_peaks(refresh=True)

    def toggle_errorbar(self, line_index, line_options):
        container = self._canvas.figure.gca().containers[line_index]
        error_bar_elements = container.get_children()[1:]

        if not line_options['error_bar'] and self.get_line_visible(line_index):
            for element in error_bar_elements:
                element.set_alpha(0)
        else:
            for element in error_bar_elements:
                element.set_alpha(1)

    def set_is_icut(self, is_icut):
        if is_icut: #disconnect quick options if icut
            self.manager.button_pressed_connected(False)
            self.manager.picking_connected(False)

        self.plot_window.action_save_cut.setVisible(is_icut)
        self.plot_window.action_plot_options.setVisible(not is_icut)
        self.plot_window.keep_make_current_seperator.setVisible(not is_icut)
        self.plot_window.action_keep.setVisible(not is_icut)
        self.plot_window.action_make_current.setVisible(not is_icut)
        self.plot_window.action_flip_axis.setVisible(is_icut)
        self.plot_window.action_gen_script.setVisible(not is_icut)
        self.plot_window.action_gen_script_clipboard.setVisible(not is_icut)
        self.plot_window.action_waterfall.setVisible(not is_icut)

        self.plot_window.showNormal()
        self.plot_window.activateWindow()
        self.plot_window.raise_()
        self._is_icut = is_icut

    def is_icut(self):
        return self._is_icut

    def _get_overplot_datum(self):
        if self._datum_dirty:
            if not self.waterfall:
                self._datum_cache = np.nanmedian([line.get_ydata() for line in self._canvas.figure.gca().get_lines()
                                                  if not self._cut_plotter_presenter.is_overplot(line)])
            else:
                for line in self._canvas.figure.gca().get_lines():
                    if not self._cut_plotter_presenter.is_overplot(line):
                        self._datum_cache = np.nanmedian([line.get_ydata()])
                        break

            self._datum_dirty = False

        return self._datum_cache

    def update_bragg_peaks(self, refresh=False):
        if self.plot_window.action_aluminium.isChecked():
            refresh and self._cut_plotter_presenter.hide_overplot_line(None, 'Aluminium')
            self._cut_plotter_presenter.add_overplot_line(self.ws_name, 'Aluminium', False, None, self.y_log,
                                                          self._get_overplot_datum())
        if self.plot_window.action_copper.isChecked():
            refresh and self._cut_plotter_presenter.hide_overplot_line(None, 'Copper')
            self._cut_plotter_presenter.add_overplot_line(self.ws_name, 'Copper', False, None, self.y_log,
                                                          self._get_overplot_datum())
        if self.plot_window.action_niobium.isChecked():
            refresh and self._cut_plotter_presenter.hide_overplot_line(None, 'Niobium')
            self._cut_plotter_presenter.add_overplot_line(self.ws_name, 'Niobium', False, None, self.y_log,
                                                          self._get_overplot_datum())
        if self.plot_window.action_tantalum.isChecked():
            refresh and self._cut_plotter_presenter.hide_overplot_line(None, 'Tantalum')
            self._cut_plotter_presenter.add_overplot_line(self.ws_name, 'Tantalum', False, None, self.y_log,
                                                          self._get_overplot_datum())
        self.update_legend()

    def save_icut(self):
        icut = self._cut_plotter_presenter.get_icut()
        return icut.save_cut()

    def flip_icut(self):
        icut = self._cut_plotter_presenter.get_icut()
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
            try:
                container = self._lines[line]
            except KeyError:
                return -1
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

    def legend_visible(self, index: int) -> bool:
        try:
            v = self._legends_visible[index]
        except IndexError:
            v = self.get_line_visible(index)
            self._legends_visible.append(v)
        return v

    def line_containers(self):
        """build dictionary of lines and their containers"""
        line_containers = {}
        containers = self._canvas.figure.gca().containers
        for container in containers:
            line = container.get_children()[0]
            line_containers[line] = container
        return line_containers

    def get_line_visible(self, line_index: int) -> bool:
        try:
            line_visible = self._lines_visible[line_index]
            return line_visible
        except KeyError:
            try:
                container = self._canvas.figure.gca().containers[line_index]
                line = container.get_children()[0]
                line_visible = line.get_visible()
            except IndexError:
                line_visible = True
            self._lines_visible[line_index] = line_visible
            return line_visible

    def toggle_waterfall(self):
        if self.waterfall:
            self._apply_offset(self.plot_window.waterfall_x, self.plot_window.waterfall_y)
        else:
            self._apply_offset(0., 0.)

        self._datum_dirty = True
        self.update_bragg_peaks(refresh=True)
        self._canvas.draw()

    def _cache_line(self, line):
        if isinstance(line, Line2D):
            self._waterfall_cache[line] = [line.get_xdata(), line.get_ydata()]
        elif isinstance(line, LineCollection):
            self._waterfall_cache[line] = [np.copy(path.vertices) for path in line._paths]

    def _apply_offset(self, x, y):
        for ind, line_containers in enumerate(self._canvas.figure.gca().containers):
            for line in line_containers.get_children():
                line not in self._waterfall_cache and self._cache_line(line)
                if isinstance(line, Line2D):
                    line.set_xdata(self._waterfall_cache[line][0] + ind * x)
                    line.set_ydata(self._waterfall_cache[line][1] + ind * y)
                elif isinstance(line, LineCollection):
                    for index, path in enumerate(line._paths):
                        if not np.isnan(path.vertices).any():
                            path.vertices = np.add(self._waterfall_cache[line][index],
                                                   np.array([[ind * x, ind * y], [ind * x, ind * y]]))

    def on_newplot(self, ax, plot_over):
        # This callback should be activated by a call to errorbar
        new_line = False
        line_containers = self._canvas.figure.gca().containers
        num_lines = len(line_containers)
        self.plot_window.action_waterfall.setEnabled(num_lines > 1)
        self.plot_window.toggle_waterfall_edit()
        if not self._is_icut and not plot_over:
            self.plot_window.action_aluminium.setChecked(False)
            self.plot_window.action_copper.setChecked(False)
            self.plot_window.action_niobium.setChecked(False)
            self.plot_window.action_tantalum.setChecked(False)
            self.plot_window.action_cif_file.setChecked(False)
        all_lines = [line for container in line_containers for line in container.get_children()]
        for cached_lines in list(self._waterfall_cache.keys()):
            if cached_lines not in all_lines:
                self._waterfall_cache.pop(cached_lines)
        for line in all_lines:
            if isinstance(line, Line2D):
                if line not in self._waterfall_cache:
                    self._waterfall_cache[line] = [line.get_xdata(), line.get_ydata()]
                    new_line = True
        if new_line and num_lines > 1:
            self.toggle_waterfall()

        self._datum_dirty = True
        self.update_bragg_peaks(refresh=True)

    def generate_script(self, clipboard=False):
        try:
            generate_script(self.ws_name, None, self, self.plot_window, clipboard)
        except Exception as e:
            # We don't want any exceptions raised in the GUI as could crash the GUI
            self.plot_window.display_error(e.message)

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
        if check_latex(value):
            self.manager.title = value
        else:
            self.plot_window.display_error("invalid latex string")

    @property
    def x_label(self):
        return self.manager.x_label

    @x_label.setter
    def x_label(self, value):
        if check_latex(value):
            self.manager.x_label = value
        else:
            self.plot_window.display_error("invalid latex string")

    @property
    def y_label(self):
        return self.manager.y_label

    @y_label.setter
    def y_label(self, value):
        if check_latex(value):
            self.manager.y_label = value
        else:
            self.plot_window.display_error("invalid latex string")

    @property
    def x_range(self):
        return self.manager.x_range

    @x_range.setter
    def x_range(self, value):
        self.manager.x_range = value

    @property
    def x_range_font_size(self):
        return self.manager.x_range_font_size

    @x_range_font_size.setter
    def x_range_font_size(self, font_size):
        self.manager.x_range_font_size = font_size

    @property
    def y_range(self):
        return self.manager.y_range

    @y_range.setter
    def y_range(self, value):
        self.manager.y_range = value

    @property
    def y_range_font_size(self):
        return self.manager.y_range_font_size

    @y_range_font_size.setter
    def y_range_font_size(self, font_size):
        self.manager.y_range_font_size = font_size

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

    @property
    def waterfall(self):
        return self.plot_window.waterfall

    @waterfall.setter
    def waterfall(self, value):
        self.plot_window.waterfall = value

    @property
    def waterfall_x(self):
        return self.plot_window.waterfall_x

    @waterfall_x.setter
    def waterfall_x(self, value):
        self.plot_window.waterfall_x = value

    @property
    def waterfall_y(self):
        return self.plot_window.waterfall_y

    @waterfall_y.setter
    def waterfall_y(self, value):
        self.plot_window.waterfall_y = value

    def is_changed(self, item):
        if self.default_options is None:
            return False
        return self.default_options[item] != getattr(self, item)
