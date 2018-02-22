from functools import partial
import six

from mslice.util.qt import QtWidgets
from mslice.util.qt.QtCore import Qt

import os.path as path
import matplotlib.colors as colors

from mantid.simpleapi import AnalysisDataService

from mslice.presenters.plot_options_presenter import SlicePlotOptionsPresenter
from mslice.presenters.quick_options_presenter import quick_options
from .interactive_cut import InteractiveCut
from .plot_options import SlicePlotOptions


class SlicePlot(object):

    def __init__(self, plot_figure, canvas, slice_plotter, new_figure):
        self.plot_figure = plot_figure
        self._canvas = canvas
        self._slice_plotter = slice_plotter
        self._ws_title = plot_figure.title
        self._arbitrary_nuclei = None
        self._cif_file = None
        self._cif_path = None
        self._quick_presenter = None
        self._legend_dict = {}
        self.icut_event = [None, None]
        self.icut = None
        if new_figure:
            self.setup_connections(plot_figure)
        self._update_lines()

    def setup_connections(self, plot_figure):
        self.plot_figure.move_window(-self.plot_figure.width() / 2, 0)

        plot_figure.actionInteractive_Cuts.setVisible(True)
        plot_figure.actionInteractive_Cuts.triggered.connect(self.interactive_cuts)
        plot_figure.actionSave_Cut.triggered.connect(self.save_icut)

        plot_figure.actionS_Q_E.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionS_Q_E,
                                                          self._slice_plotter.show_scattering_function, False))
        plot_figure.actionChi_Q_E.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionChi_Q_E,
                                                            self._slice_plotter.show_dynamical_susceptibility, True))

        plot_figure.actionChi_Q_E_magnetic.triggered.connect(
            partial(self.show_intensity_plot, plot_figure.actionChi_Q_E_magnetic,
                    self._slice_plotter.show_dynamical_susceptibility_magnetic, True))

        plot_figure.actionD2sigma_dOmega_dE.triggered.connect(
            partial(self.show_intensity_plot, plot_figure.actionD2sigma_dOmega_dE,
                    self._slice_plotter.show_d2sigma, False))

        plot_figure.actionSymmetrised_S_Q_E.triggered.connect(
            partial(self.show_intensity_plot, plot_figure.actionSymmetrised_S_Q_E,
                    self._slice_plotter.show_symmetrised, True))

        plot_figure.actionGDOS.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionGDOS,
                                                         self._slice_plotter.show_gdos, True))

        plot_figure.actionHydrogen.triggered.connect(
            partial(self.toggle_overplot_line, plot_figure.actionHydrogen, 1, True))
        plot_figure.actionDeuterium.triggered.connect(
            partial(self.toggle_overplot_line, plot_figure.actionDeuterium, 2, True))
        plot_figure.actionHelium.triggered.connect(
            partial(self.toggle_overplot_line, plot_figure.actionHelium, 4, True))
        plot_figure.actionArbitrary_nuclei.triggered.connect(self.arbitrary_recoil_line)
        plot_figure.actionAluminium.triggered.connect(
            partial(self.toggle_overplot_line, plot_figure.actionAluminium, 'Aluminium', False))
        plot_figure.actionCopper.triggered.connect(
            partial(self.toggle_overplot_line, plot_figure.actionCopper, 'Copper', False))
        plot_figure.actionNiobium.triggered.connect(
            partial(self.toggle_overplot_line, plot_figure.actionNiobium, 'Niobium', False))
        plot_figure.actionTantalum.triggered.connect(
            partial(self.toggle_overplot_line, plot_figure.actionTantalum, 'Tantalum', False))
        plot_figure.actionCIF_file.triggered.connect(partial(self.cif_file_powder_line))

    def plot_options(self):
        new_config = SlicePlotOptionsPresenter(SlicePlotOptions(), self).get_new_config()
        if new_config:
            self._canvas.draw()

    def plot_clicked(self, x, y):
        bounds = self.calc_figure_boundaries()
        if bounds['x_label'] < y < bounds['title']:
            if bounds['y_label'] < x < bounds['colorbar_label']:
                if y < bounds['x_range']:
                    self._quick_presenter = quick_options('x_range', self)
                elif x < bounds['y_range']:
                    self._quick_presenter = quick_options('y_range', self)
                elif x > bounds['colorbar_range']:
                    self._quick_presenter = quick_options('colorbar_range', self, self.colorbar_log)
            self._canvas.draw()

    def object_clicked(self, target):
        if target in self._legend_dict:
            self._quick_presenter = quick_options(self._legend_dict[target], self)
        else:
            self._quick_presenter = quick_options(target, self)
        self.reset_info_checkboxes()
        self.update_legend()
        self._canvas.draw()

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

    def change_axis_scale(self, colorbar_range, logarithmic):
        current_axis = self._canvas.figure.gca()
        images = current_axis.get_images()
        if len(images) != 1:
            raise RuntimeError("Expected single image on axes, found " + str(len(images)))
        mappable = images[0]
        vmin, vmax = colorbar_range
        if logarithmic and type(mappable.norm) != colors.LogNorm:
            mappable.colorbar.remove()
            if vmin == float(0):
                vmin = 0.001
            norm = colors.LogNorm(vmin, vmax)
            mappable.set_norm(norm)
            self._canvas.figure.colorbar(mappable)
        elif not logarithmic and type(mappable.norm) != colors.Normalize:
            mappable.colorbar.remove()
            norm = colors.Normalize(vmin, vmax)
            mappable.set_norm(norm)
            self._canvas.figure.colorbar(mappable)
        mappable.set_clim((vmin, vmax))

    def reset_info_checkboxes(self):
        for key, line in six.iteritems(self._slice_plotter.overplot_lines[self._ws_title]):
            if str(line.get_linestyle()) == 'None':
                if isinstance(key, int):
                    key = self._slice_plotter.get_recoil_label(key)
                action_checked = getattr(self, 'action' + key)
                action_checked.setChecked(False)

    def toggle_overplot_line(self, action, key, recoil, checked, cif_file=None):
        if checked:
            self._slice_plotter.add_overplot_line(self._ws_title, key, recoil, cif_file)
        else:
            self._slice_plotter.hide_overplot_line(self._ws_title, key)
        self.update_legend()
        self._canvas.draw()

    def arbitrary_recoil_line(self):
        checked = self.plot_figure.actionArbitrary_nuclei.isChecked()
        if checked:
            self._arbitrary_nuclei, confirm = QtWidgets.QInputDialog.getInt(self.plot_figure, 'Arbitrary Nuclei', 'Enter relative mass:')
            if not confirm:
                return
        self.toggle_overplot_line(self.plot_figure.actionArbitrary_nuclei, self._arbitrary_nuclei, True, checked)

    def cif_file_powder_line(self, checked):
        if checked:
            cif_path = QtWidgets.QFileDialog().getOpenFileName(self.plot_figure, 'Open CIF file', '/home', 'Files (*.cif)')
            cif_path = str(cif_path[0]) if isinstance(cif_path, tuple) else str(cif_path)
            key = path.basename(cif_path).rsplit('.')[0]
            self._cif_file = key
            self._cif_path = cif_path
        else:
            key = self._cif_file
            cif_path = None
        if key:
            self.toggle_overplot_line(self.plot_figure.actionCIF_file, key, False,
                                      self.plot_figure.actionCIF_file.isChecked(), cif_file=cif_path)
        else:
            self.plot_figure.actionCIF_file.setChecked(False)

    def update_legend(self):
        lines = []
        axes = self._canvas.figure.gca()
        for line in axes.get_lines():
            if str(line.get_linestyle()) != 'None' and line.get_label() != '':
                lines.append(line)
        if len(lines) > 0:
            legend = axes.legend(fontsize='small')
            for legline, line in zip(legend.get_lines(), lines):
                legline.set_picker(5)
                self._legend_dict[legline] = line
            for label, line in zip(legend.get_texts(), lines):
                label.set_picker(5)
                self._legend_dict[label] = line
        else:
            axes.legend_ = None  # remove legend

    def intensity_selection(self, selected):
        '''Ticks selected and un-ticks other intensity options. Returns previous selection'''
        options = self.plot_figure.menuIntensity.actions()
        previous = None
        for op in options:
            if op.isChecked() and op is not selected:
                previous = op
            op.setChecked(False)
        selected.setChecked(True)
        return previous

    def show_intensity_plot(self, action, slice_plotter_method, temp_dependent):
        if action.isChecked():
            previous = self.intensity_selection(action)
            cbar_log = self.colorbar_log
            x_range = self.x_range
            y_range = self.y_range
            title = self.title
            if temp_dependent:
                if not self._run_temp_dependent(slice_plotter_method, previous):
                    return
            else:
                slice_plotter_method(self._ws_title)
            self.change_axis_scale(self.colorbar_range, cbar_log)
            self.x_range = x_range
            self.y_range = y_range
            self.title = title
            self.plot_figure.update_grid()
            self._update_lines()
            self._canvas.draw()
        else:
            action.setChecked(True)

    def _run_temp_dependent(self, slice_plotter_method, previous):
        try:
            slice_plotter_method(self._ws_title)
        except ValueError:  # sample temperature not yet set
            try:
                temp_value, field = self.ask_sample_temperature_field(str(self._ws_title))
            except RuntimeError:  # if cancel is clicked, go back to previous selection
                self.intensity_selection(previous)
                return False
            if field:
                self._slice_plotter.add_sample_temperature_field(temp_value)
                self._slice_plotter.update_sample_temperature(self._ws_title)
            else:
                try:
                    temp_value = float(temp_value)
                    if temp_value < 0:
                        raise ValueError
                except ValueError:
                    self.plot_figure.error_box("Invalid value entered for sample temperature. Enter a value in Kelvin \
                                               or a sample log field.")
                    self.intensity_selection(previous)
                    return False
                else:
                    self._slice_plotter.set_sample_temperature(self._ws_title, temp_value)
            slice_plotter_method(self._ws_title)
        return True

    def ask_sample_temperature_field(self, ws_name):
        ws_name = ws_name[:ws_name.rfind("_")]
        ws = AnalysisDataService[ws_name]
        temp_field, confirm = QtWidgets.QInputDialog.getItem(self.plot_figure, 'Sample Temperature',
                                                             'Sample Temperature not found. Select the sample ' +
                                                             'temperature field or enter a value in Kelvin:',
                                                             ws.run().keys())
        if not confirm:
            raise RuntimeError("sample_temperature_dialog cancelled")
        else:
            return str(temp_field), temp_field in ws.run().keys()

    def _update_lines(self):
        """ Updates the powder/recoil overplots lines when intensity type changes """
        lines = {self.plot_figure.actionHydrogen:[1, True, ''],
                 self.plot_figure.actionDeuterium:[2, True, ''],
                 self.plot_figure.actionHelium:[4, True, ''],
                 self.plot_figure.actionArbitrary_nuclei:[self._arbitrary_nuclei, True, ''],
                 self.plot_figure.actionAluminium:['Aluminium', False, ''],
                 self.plot_figure.actionCopper:['Copper', False, ''],
                 self.plot_figure.actionNiobium:['Niobium', False, ''],
                 self.plot_figure.actionTantalum:['Tantalum', False, ''],
                 self.plot_figure.actionCIF_file:[self._cif_file, False, self._cif_path]}
        for line in lines:
            if line.isChecked():
                if  lines[line][0] in self._slice_plotter.overplot_lines[self._ws_title]:
                    self._slice_plotter.overplot_lines[self._ws_title].pop(lines[line][0])
                self._slice_plotter.add_overplot_line(self._ws_title, *lines[line])
                self.update_legend()

    def get_line_data(self, target):
        line_options = {}
        line_options['label'] = target.get_label()
        line_options['legend'] = None
        line_options['shown'] = None
        line_options['color'] = target.get_color()
        line_options['style'] = target.get_linestyle()
        line_options['width'] = str(int(target.get_linewidth()))
        line_options['marker'] = target.get_marker()
        return line_options

    def set_line_data(self, line, line_options):
        line.set_label(line_options['label'])
        line.set_linestyle(line_options['style'])
        line.set_marker(line_options['marker'])
        line.set_color(line_options['color'])
        line.set_linewidth(line_options['width'])

    def interactive_cuts(self):
        if not self.icut:
            self.plot_figure.picking_connected(False)
            self.plot_figure.actionKeep.trigger()
            self.plot_figure.actionKeep.setEnabled(False)
            self.plot_figure.actionMakeCurrent.setEnabled(False)
            self.plot_figure.actionSave_Cut.setVisible(True)
            self._canvas.setCursor(Qt.CrossCursor)
        else:
            self.plot_figure.picking_connected(True)
            self.plot_figure.actionKeep.setEnabled(True)
            self.plot_figure.actionMakeCurrent.setEnabled(True)
            self.plot_figure.actionSave_Cut.setVisible(False)
            self._canvas.setCursor(Qt.ArrowCursor)
        self.toggle_icut()

    def toggle_icut(self):
        if self.icut is not None:
            self.icut.clear()
            self.icut = None
        else:
            self.icut = InteractiveCut(self, self._canvas, self._ws_title)

    def save_icut(self):
        self.icut.save_cut()

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
        return self._canvas.figure.gca().get_images()[0].get_clim()

    @colorbar_range.setter
    def colorbar_range(self, value):
        self.change_axis_scale(value, self.colorbar_log)

    @property
    def colorbar_log(self):
        mappable = self._canvas.figure.gca().get_images()[0]
        norm = mappable.norm
        return isinstance(norm, colors.LogNorm)

    @colorbar_log.setter
    def colorbar_log(self, value):
        self.change_axis_scale(self.colorbar_range, value)

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

    @property
    def x_grid(self):
        return self.plot_figure.x_grid

    @x_grid.setter
    def x_grid(self, value):
        self.plot_figure.x_grid = value

    @property
    def y_grid(self):
        return self.plot_figure.y_grid

    @y_grid.setter
    def y_grid(self, value):
        self.plot_figure.y_grid = value
