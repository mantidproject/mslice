from functools import partial
import six

from mslice.util.qt import QtWidgets
from mslice.util.qt.QtCore import Qt

import os.path as path
import matplotlib.colors as colors
from matplotlib.lines import Line2D

from mslice.presenters.plot_options_presenter import SlicePlotOptionsPresenter
from mslice.presenters.quick_options_presenter import quick_options
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.plotting.plot_window.iplot import IPlot
from mslice.plotting.plot_window.interactive_cut import InteractiveCut
from mslice.plotting.plot_window.plot_options import SlicePlotOptions


class SlicePlot(IPlot):

    def __init__(self, figure_manager, slice_plotter, workspace_name):
        self.manager = figure_manager
        self.plot_window = figure_manager.window
        self._canvas = self.plot_window.canvas
        self._slice_plotter = slice_plotter
        self.ws_name = workspace_name
        self._arb_nuclei_rmm = None
        self._cif_file = None
        self._cif_path = None
        self._legend_dict = {}
        self.icut_event = [None, None]
        self.icut = None
        self.setup_connections(self.plot_window)

    def setup_connections(self, plot_figure):
        plot_figure.action_interactive_cuts.setVisible(True)
        plot_figure.action_interactive_cuts.triggered.connect(self.interactive_cuts)
        plot_figure.action_save_cut.setVisible(False)
        plot_figure.action_save_cut.triggered.connect(self.save_icut)
        plot_figure.action_flip_axis.setVisible(False)
        plot_figure.action_flip_axis.triggered.connect(self.flip_icut)

        plot_figure.action_sqe.triggered.connect(partial(self.show_intensity_plot, plot_figure.action_sqe,
                                                         self._slice_plotter.show_scattering_function, False))
        plot_figure.action_chi_qe.triggered.connect(partial(self.show_intensity_plot, plot_figure.action_chi_qe,
                                                            self._slice_plotter.show_dynamical_susceptibility, True))

        plot_figure.action_chi_qe_magnetic.triggered.connect(
            partial(self.show_intensity_plot, plot_figure.action_chi_qe_magnetic,
                    self._slice_plotter.show_dynamical_susceptibility_magnetic, True))

        plot_figure.action_d2sig_dw_de.triggered.connect(
            partial(self.show_intensity_plot, plot_figure.action_d2sig_dw_de,
                    self._slice_plotter.show_d2sigma, False))

        plot_figure.action_symmetrised_sqe.triggered.connect(
            partial(self.show_intensity_plot, plot_figure.action_symmetrised_sqe,
                    self._slice_plotter.show_symmetrised, True))

        plot_figure.action_gdos.triggered.connect(partial(self.show_intensity_plot, plot_figure.action_gdos,
                                                          self._slice_plotter.show_gdos, True))

        plot_figure.action_hydrogen.triggered.connect(
            partial(self.toggle_overplot_line, 1, True))
        plot_figure.action_deuterium.triggered.connect(
            partial(self.toggle_overplot_line, 2, True))
        plot_figure.action_helium.triggered.connect(
            partial(self.toggle_overplot_line, 4, True))
        plot_figure.action_arbitrary_nuclei.triggered.connect(self.arbitrary_recoil_line)
        plot_figure.action_aluminium.triggered.connect(
            partial(self.toggle_overplot_line, 'Aluminium', False))
        plot_figure.action_copper.triggered.connect(
            partial(self.toggle_overplot_line, 'Copper', False))
        plot_figure.action_niobium.triggered.connect(
            partial(self.toggle_overplot_line, 'Niobium', False))
        plot_figure.action_tantalum.triggered.connect(
            partial(self.toggle_overplot_line, 'Tantalum', False))
        plot_figure.action_cif_file.triggered.connect(partial(self.cif_file_powder_line))

    def disconnect(self, plot_window):
        plot_window.action_interactive_cuts.triggered.disconnect()
        plot_window.action_save_cut.triggered.disconnect()
        plot_window.action_flip_axis.triggered.disconnect()
        plot_window.action_sqe.triggered.disconnect()
        plot_window.action_chi_qe.triggered.disconnect()
        plot_window.action_chi_qe_magnetic.triggered.disconnect()
        plot_window.action_d2sig_dw_de.triggered.disconnect()
        plot_window.action_symmetrised_sqe.triggered.disconnect()
        plot_window.action_gdos.triggered.disconnect()
        plot_window.action_hydrogen.triggered.disconnect()
        plot_window.action_deuterium.triggered.disconnect()
        plot_window.action_helium.triggered.disconnect()
        plot_window.action_arbitrary_nuclei.triggered.disconnect()
        plot_window.action_aluminium.triggered.disconnect()
        plot_window.action_copper.triggered.disconnect()
        plot_window.action_niobium.triggered.disconnect()
        plot_window.action_tantalum.triggered.disconnect()
        plot_window.action_cif_file.triggered.disconnect()

    def window_closing(self):
        # nothing to do
        pass

    def plot_options(self):
        new_config = SlicePlotOptionsPresenter(SlicePlotOptions(), self).get_new_config()
        if new_config:
            self._canvas.draw()

    def plot_clicked(self, x, y):
        bounds = self.calc_figure_boundaries()
        if bounds['x_label'] < y < bounds['title']:
            if bounds['y_label'] < x < bounds['colorbar_label']:
                if y < bounds['x_range']:
                    quick_options('x_range', self)
                elif x < bounds['y_range']:
                    quick_options('y_range', self)
                elif x > bounds['colorbar_range']:
                    quick_options('colorbar_range', self, self.colorbar_log)
            self._canvas.draw()

    def object_clicked(self, target):
        if target in self._legend_dict:
            quick_options(self._legend_dict[target], self)
        else:
            quick_options(target, self)
        self.update_legend()
        self._canvas.draw()

    def update_legend(self):
        lines = []
        labels = []
        axes = self._canvas.figure.gca()
        line_artists = [artist for artist in axes.get_children() if isinstance(artist, Line2D)]
        for line in line_artists:
            if str(line.get_linestyle()) != 'None' and line.get_label() != '':
                lines.append(line)
                labels.append(line.get_label())
        if len(lines) > 0:
            legend = axes.legend(lines, labels, fontsize='small')
            for legline, line in zip(legend.get_lines(), lines):
                legline.set_picker(5)
                self._legend_dict[legline] = line
            for label, line in zip(legend.get_texts(), lines):
                label.set_picker(5)
                self._legend_dict[label] = line
        else:
            axes.legend_ = None  # remove legend

    def change_axis_scale(self, colorbar_range, logarithmic):
        current_axis = self._canvas.figure.gca()
        colormesh = current_axis.collections[0]
        vmin, vmax = colorbar_range
        if logarithmic:
            label = self.colorbar_label
            colormesh.colorbar.remove()
            if vmin <= float(0):
                vmin = 0.001
            colormesh.set_clim((vmin, vmax))
            norm = colors.LogNorm(vmin, vmax)
            colormesh.set_norm(norm)
            self._canvas.figure.colorbar(colormesh)
            self.colorbar_label = label
        else:
            label = self.colorbar_label
            colormesh.colorbar.remove()
            colormesh.set_clim((vmin, vmax))
            norm = colors.Normalize(vmin, vmax)
            colormesh.set_norm(norm)
            self._canvas.figure.colorbar(colormesh)
            self.colorbar_label = label

    def get_line_options(self, target):
        line_options = {}
        line_options['label'] = target.get_label()
        line_options['legend'] = None
        line_options['shown'] = None
        line_options['color'] = target.get_color()
        line_options['style'] = target.get_linestyle()
        line_options['width'] = str(int(target.get_linewidth()))
        line_options['marker'] = target.get_marker()
        return line_options

    def set_line_options(self, line, line_options):
        line.set_label(line_options['label'])
        line.set_linestyle(line_options['style'])
        line.set_marker(line_options['marker'])
        line.set_color(line_options['color'])
        line.set_linewidth(line_options['width'])

    def calc_figure_boundaries(self):
        fig_x, fig_y = self._canvas.figure.get_size_inches() * self._canvas.figure.dpi
        bounds = {}
        bounds['y_label'] = fig_x * 0.07
        bounds['y_range'] = fig_x * 0.12
        bounds['colorbar_range'] = fig_x * 0.75
        bounds['colorbar_label'] = fig_x * 0.86
        bounds['title'] = fig_y * 0.9
        bounds['x_range'] = fig_y * 0.09
        bounds['x_label'] = fig_y * 0.05
        return bounds

    def toggle_overplot_line(self, key, recoil, checked, cif_file=None):
        if checked:
            self._slice_plotter.add_overplot_line(self.ws_name, key, recoil, cif_file)
        else:
            self._slice_plotter.hide_overplot_line(self.ws_name, key)
        self.update_legend()
        self._canvas.draw()

    def arbitrary_recoil_line(self):
        recoil = True
        checked = self.plot_window.action_arbitrary_nuclei.isChecked()
        if checked:
            self._arb_nuclei_rmm, confirm = QtWidgets.QInputDialog.getInt(
                self.plot_window, 'Arbitrary Nuclei', 'Enter relative mass:')
            if confirm:
                self.toggle_overplot_line(self._arb_nuclei_rmm, recoil, checked)
        else:
            self.toggle_overplot_line(self._arb_nuclei_rmm, recoil, checked)

    def cif_file_powder_line(self, checked):
        if checked:
            cif_path = QtWidgets.QFileDialog().getOpenFileName(self.plot_window, 'Open CIF file', '/home', 'Files (*.cif)')
            cif_path = str(cif_path[0]) if isinstance(cif_path, tuple) else str(cif_path)
            key = path.basename(cif_path).rsplit('.')[0]
            self._cif_file = key
            self._cif_path = cif_path
        else:
            key = self._cif_file
            cif_path = None
        if key:
            recoil = False
            self.toggle_overplot_line(key, recoil, checked, cif_file=cif_path)

    def _reset_intensity(self):
        options = self.plot_window.menu_intensity.actions()
        for op in options:
            op.setChecked(False)

    def selected_intensity(self):
        options = self.plot_window.menu_intensity.actions()
        for op in options:
            if op.isChecked():
                return op

    def set_intensity(self, intensity):
        self._reset_intensity()
        intensity.setChecked(True)

    def show_intensity_plot(self, action, slice_plotter_method, temp_dependent):
        if action.isChecked():
            previous = self.selected_intensity()
            self.set_intensity(action)
            cbar_log = self.colorbar_log
            cbar_range = self.colorbar_range
            x_range = self.x_range
            y_range = self.y_range
            title = self.title
            if temp_dependent:
                if not self._run_temp_dependent(slice_plotter_method, previous):
                    return
            else:
                slice_plotter_method(self.ws_name)
            self.change_axis_scale(cbar_range, cbar_log)
            self.x_range = x_range
            self.y_range = y_range
            self.title = title
            self.manager.update_grid()
            self._update_lines()
            self._canvas.draw()
        else:
            action.setChecked(True)

    def _run_temp_dependent(self, slice_plotter_method, previous):
        try:
            slice_plotter_method(self.ws_name)
        except ValueError:  # sample temperature not yet set
            try:
                temp_value, field = self.ask_sample_temperature_field(str(self.ws_name))
            except RuntimeError:  # if cancel is clicked, go back to previous selection
                self.set_intensity(previous)
                return False
            if field:
                self._slice_plotter.add_sample_temperature_field(temp_value)
                self._slice_plotter.update_sample_temperature(self.ws_name)
            else:
                try:
                    temp_value = float(temp_value)
                    if temp_value < 0:
                        raise ValueError
                except ValueError:
                    self.plot_window.error_box("Invalid value entered for sample temperature. Enter a value in Kelvin \
                                               or a sample log field.")
                    self.set_intensity(previous)
                    return False
                else:
                    self._slice_plotter.set_sample_temperature(self.ws_name, temp_value)
            slice_plotter_method(self.ws_name)
        return True

    def ask_sample_temperature_field(self, ws_name):
        ws = get_workspace_handle(ws_name)
        try:
            keys = ws.raw_ws.run().keys()
        except AttributeError:
            keys = ws.raw_ws.getExperimentInfo(0).run().keys()
        temp_field, confirm = QtWidgets.QInputDialog.getItem(self.plot_window, 'Sample Temperature',
                                                             'Sample Temperature not found. Select the sample ' +
                                                             'temperature field or enter a value in Kelvin:',
                                                             keys)
        if not confirm:
            raise RuntimeError("sample_temperature_dialog cancelled")
        else:
            return str(temp_field), temp_field in keys

    def _update_lines(self):
        """ Updates the powder/recoil overplots lines when intensity type changes """
        lines = {self.plot_window.action_hydrogen:[1, True, ''],
                 self.plot_window.action_deuterium:[2, True, ''],
                 self.plot_window.action_helium:[4, True, ''],
                 self.plot_window.action_arbitrary_nuclei:[self._arb_nuclei_rmm, True, ''],
                 self.plot_window.action_aluminium:['Aluminium', False, ''],
                 self.plot_window.action_copper:['Copper', False, ''],
                 self.plot_window.action_niobium:['Niobium', False, ''],
                 self.plot_window.action_tantalum:['Tantalum', False, ''],
                 self.plot_window.action_cif_file:[self._cif_file, False, self._cif_path]}
        for line in lines:
            if line.isChecked():
                self._slice_plotter.add_overplot_line(self.ws_name, *lines[line])
        self.update_legend()
        self._canvas.draw()

    def interactive_cuts(self):
        if not self.icut:
            self.manager.picking_connected(False)
            if self.plot_window.action_zoom_in.isChecked():
                self.plot_window.action_zoom_in.setChecked(False)
                self.plot_window.action_zoom_in.triggered.emit(False)  # turn off zoom
            self.plot_window.action_zoom_in.setEnabled(False)
            self.plot_window.action_keep.trigger()
            self.plot_window.action_keep.setEnabled(False)
            self.plot_window.action_make_current.setEnabled(False)
            self.plot_window.action_save_cut.setVisible(True)
            self.plot_window.action_flip_axis.setVisible(True)
            self._canvas.setCursor(Qt.CrossCursor)
        else:
            self.manager.picking_connected(True)
            self.plot_window.action_zoom_in.setEnabled(True)
            self.plot_window.action_keep.setEnabled(True)
            self.plot_window.action_make_current.setEnabled(True)
            self.plot_window.action_save_cut.setVisible(False)
            self.plot_window.action_flip_axis.setVisible(False)
            self._canvas.setCursor(Qt.ArrowCursor)
        self.toggle_icut()

    def toggle_icut(self):
        if self.icut is not None:
            self.icut.clear()
            self.icut = None
        else:
            self.icut = InteractiveCut(self, self._canvas, self.ws_name)

    def save_icut(self):
        self.icut.save_cut()

    def flip_icut(self):
        self.icut.flip_axis()

    def update_workspaces(self):
        self._slice_plotter.update_displayed_workspaces()


    @property
    def colorbar_label(self):
        return self._canvas.figure.get_axes()[1].get_ylabel()

    @colorbar_label.setter
    def colorbar_label(self, value):
        self._canvas.figure.get_axes()[1].set_ylabel(value, labelpad=20, rotation=270, picker=5)

    @property
    def colorbar_range(self):
        return self._canvas.figure.gca().collections[0].get_clim()

    @colorbar_range.setter
    def colorbar_range(self, value):
        self.change_axis_scale(value, self.colorbar_log)

    @property
    def colorbar_log(self):
        return isinstance(self._canvas.figure.gca().collections[0].norm, colors.LogNorm)

    @colorbar_log.setter
    def colorbar_log(self, value):
        self.change_axis_scale(self.colorbar_range, value)

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
