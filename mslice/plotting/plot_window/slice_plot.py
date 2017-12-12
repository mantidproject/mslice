from functools import partial
import six
from PyQt4 import QtGui
import os.path as path
import matplotlib.colors as colors

from mantid.simpleapi import AnalysisDataService

from mslice.presenters.plot_options_presenter import SlicePlotOptionsPresenter
from .plot_options import SlicePlotOptions


class SlicePlot(object):
    
    def __init__(self, plot_figure, canvas, slice_plotter):
        self.plot_figure = plot_figure
        self.canvas = canvas
        self.slice_plotter = slice_plotter
        self.ws_title = plot_figure.title
        self.arbitrary_nuclei = None
        self.cif_file = None
    
        plot_figure.actionS_Q_E.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionS_Q_E,
                                                   self.slice_plotter.show_scattering_function, False))
        plot_figure.actionChi_Q_E.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionChi_Q_E,
                                                     self.slice_plotter.show_dynamical_susceptibility, True))
        plot_figure.actionChi_Q_E_magnetic.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionChi_Q_E_magnetic,
                                                              self.slice_plotter.show_dynamical_susceptibility_magnetic,
                                                              True))
        plot_figure.actionD2sigma_dOmega_dE.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionD2sigma_dOmega_dE,
                                                               self.slice_plotter.show_d2sigma, False))
        plot_figure.actionSymmetrised_S_Q_E.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionSymmetrised_S_Q_E,
                                                               self.slice_plotter.show_symmetrised, True))
        plot_figure.actionGDOS.triggered.connect(partial(self.show_intensity_plot, plot_figure.actionGDOS,
                                                  self.slice_plotter.show_gdos, True))
    
        plot_figure.actionHydrogen.triggered.connect(partial(self.toggle_overplot_line, plot_figure.actionHydrogen, 1, True))
        plot_figure.actionDeuterium.triggered.connect(partial(self.toggle_overplot_line, plot_figure.actionDeuterium, 2, True))
        plot_figure.actionHelium.triggered.connect(partial(self.toggle_overplot_line, plot_figure.actionHelium, 4, True))
        plot_figure.actionArbitrary_nuclei.triggered.connect(self.arbitrary_recoil_line)
        plot_figure.actionAluminium.triggered.connect(partial(self.toggle_overplot_line, plot_figure.actionAluminium,
                                                       'Aluminium', False))
        plot_figure.actionCopper.triggered.connect(partial(self.toggle_overplot_line, plot_figure.actionCopper,
                                                    'Copper', False))
        plot_figure.actionNiobium.triggered.connect(partial(self.toggle_overplot_line, plot_figure.actionNiobium,
                                                     'Niobium', False))
        plot_figure.actionTantalum.triggered.connect(partial(self.toggle_overplot_line, plot_figure.actionTantalum,
                                                      'Tantalum', False))
        plot_figure.actionCIF_file.triggered.connect(partial(self.cif_file_powder_line))

    def plot_options(self):
        new_config = SlicePlotOptionsPresenter(SlicePlotOptions(), self).get_new_config()
        if new_config:
            self.canvas.draw()

    def post_click_event(self):
        self.reset_info_checkboxes()
        self.update_legend()

    def change_slice_plot(self, colorbar_range, logarithmic):
        current_axis = self.canvas.figure.gca()
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
            self.canvas.figure.colorbar(mappable)
        elif not logarithmic and type(mappable.norm) != colors.Normalize:
            mappable.colorbar.remove()
            norm = colors.Normalize(vmin, vmax)
            mappable.set_norm(norm)
            self.canvas.figure.colorbar(mappable)
        mappable.set_clim((vmin, vmax))

    def reset_info_checkboxes(self):
        for key, line in six.iteritems(self.slice_plotter.overplot_lines[self.ws_title]):
            if str(line.get_linestyle()) == 'None':
                if isinstance(key, int):
                    key = self.slice_plotter.get_recoil_label(key)
                action_checked = getattr(self, 'action' + key)
                action_checked.setChecked(False)

    def toggle_overplot_line(self, action, key, recoil, checked, cif_file=None):
        if checked:
            self.slice_plotter.add_overplot_line(self.ws_title, key, recoil, cif_file)
        else:
            self.slice_plotter.hide_overplot_line(self.ws_title, key)
        self.update_legend()
        self.canvas.draw()

    def arbitrary_recoil_line(self):
        if self.plot_figure.actionArbitrary_nuclei.isChecked():
            self.arbitrary_nuclei, confirm = QtGui.QInputDialog.getInt(self, 'Arbitrary Nuclei', 'Enter relative mass:')
            if not confirm:
                return
        self.toggle_overplot_line(self.plot_figure.actionArbitrary_nuclei, self.arbitrary_nuclei, True)

    def cif_file_powder_line(self, checked):
        if checked:
            cif_path = str(QtGui.QFileDialog().getOpenFileName(self, 'Open CIF file', '/home', 'Files (*.cif)'))
            key = path.basename(cif_path).rsplit('.')[0]
            self.cif_file = key
        else:
            key = self.cif_file
            cif_path = None
        self.toggle_overplot_line(self.plot_figure.actionCIF_file, key, False,
                                  self.plot_figure.actionCIF_file.isChecked(), cif_file=cif_path)

    def update_legend(self):
        lines = []
        axes = self.canvas.figure.gca()
        for line in axes.get_lines():
            if str(line.get_linestyle()) != 'None' and line.get_label() != '':
                lines.append(line)
        if len(lines) > 0:
            legend = axes.legend(fontsize='small')
            for legline, line in zip(legend.get_lines(), lines):
                legline.set_picker(5)
                self.plot_figure.legend_dict[legline] = line
            for label, line in zip(legend.get_texts(), lines):
                label.set_picker(5)
                self.plot_figure.legend_dict[label] = line
        else:
            axes.legend_ = None  # remove legend

    def toggle_legend(self):
        axes = self.canvas.figure.gca()
        if axes.legend_ is None:
            self.update_legend()
        else:
            axes.legend_ = None

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
                slice_plotter_method(self.ws_title)
            self.change_slice_plot(self.colorbar_range, cbar_log)
            self.x_range = x_range
            self.y_range = y_range
            self.title = title
            self.canvas.draw()
        else:
            action.setChecked(True)

    def _run_temp_dependent(self, slice_plotter_method, previous):
        try:
            slice_plotter_method(self.ws_title)
        except ValueError:  # sample temperature not yet set
            try:
                field = self.ask_sample_temperature_field(str(self.ws_title))
            except RuntimeError:  # if cancel is clicked, go back to previous selection
                self.intensity_selection(previous)
                return False
            self.slice_plotter.add_sample_temperature_field(field)
            self.slice_plotter.update_sample_temperature(self.ws_title)
            slice_plotter_method(self.ws_title)
        return True

    def ask_sample_temperature_field(self, ws_name):
        if ws_name[-3:] == '_QE':
            ws_name = ws_name[:-3]
        ws = AnalysisDataService[ws_name]
        temp_field, confirm = QtGui.QInputDialog.getItem(self.plot_figure, 'Sample Temperature',
                                                         'Sample Temperature not found. ' +
                                                         'Select the sample temperature field:',
                                                         ws.run().keys(), False)
        if not confirm:
            raise RuntimeError("sample_temperature_dialog cancelled")
        else:
            return str(temp_field)

    @property
    def colorbar_label(self):
        return self.canvas.figure.get_axes()[1].get_ylabel()

    @colorbar_label.setter
    def colorbar_label(self, value):
        self.canvas.figure.get_axes()[1].set_ylabel(value, labelpad=20, rotation=270, picker=5)

    @property
    def colorbar_range(self):
        return self.canvas.figure.gca().get_images()[0].get_clim()

    @colorbar_range.setter
    def colorbar_range(self, value):
        self.change_slice_plot(value, self.colorbar_log)

    @property
    def colorbar_log(self):
        mappable = self.canvas.figure.gca().get_images()[0]
        norm = mappable.norm
        return isinstance(norm, colors.LogNorm)

    @colorbar_log.setter
    def colorbar_log(self, value):
        self.change_slice_plot(self.colorbar_range, value)

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
