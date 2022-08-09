from qtpy.QtWidgets import QFileDialog

import os.path as path

from mslice.models.labels import get_recoil_label
import mslice.plotting.pyplot as plt

OVERPLOT_COLORS = {1: 'b', 2: 'g', 4: 'r', 'Aluminium': 'g',
                   'Copper': 'm', 'Niobium': 'y', 'Tantalum': 'b'}
PICKER_TOL_PTS = 5


def _update_overplot_lines(plotter_presenter, ws_name, lines):
    for line in lines:
        if line.isChecked():
            plotter_presenter.add_overplot_line(ws_name, *lines[line])


def _update_powder_lines(plot_handler, plotter_presenter):
    """ Updates the powder overplots lines when intensity type changes """
    lines = {plot_handler.plot_window.action_aluminium: ['Aluminium', False, ''],
             plot_handler.plot_window.action_copper: ['Copper', False, ''],
             plot_handler.plot_window.action_niobium: ['Niobium', False, ''],
             plot_handler.plot_window.action_tantalum: ['Tantalum', False, ''],
             plot_handler.plot_window.action_cif_file: [plot_handler._cif_file,
                                                        False, plot_handler._cif_path]}
    _update_overplot_lines(plotter_presenter, plot_handler.ws_name, lines)


def toggle_overplot_line(plot_handler, plotter_presenter, key, recoil, checked, cif_file=None):
    last_active_figure_number = None
    if plot_handler.manager._current_figs._active_figure is not None:
        last_active_figure_number = plot_handler.manager._current_figs.get_active_figure().number

    disable_make_current_after_plot = False
    if plot_handler.manager.make_current_disabled():
        plot_handler.manager.enable_make_current()
        disable_make_current_after_plot = True
    plot_handler.manager.report_as_current()

    if checked:
        intensity_correction = plot_handler.intensity_method if not plot_handler.intensity_method else \
            plot_handler.intensity_method[5:]
        plotter_presenter.add_overplot_line(plot_handler.ws_name, key, recoil, cif_file, plot_handler.y_log,
                                            plot_handler._get_overplot_datum(), intensity_correction)
    else:
        plotter_presenter.hide_overplot_line(plot_handler.ws_name, key)

    plot_handler.update_legend()
    plot_handler._canvas.draw()

    # Reset current active figure
    _reset_current_figure(plot_handler.manager, last_active_figure_number, disable_make_current_after_plot)


def _reset_current_figure(manager, last_active_figure_number, disable_make_current_after_plot):
    if last_active_figure_number is not None:
        manager._current_figs.set_figure_as_current(last_active_figure_number)
    if disable_make_current_after_plot:
        manager.disable_make_current()


def cif_file_powder_line(plot_handler, plotter_presenter, checked):
    if checked:
        cif_path = QFileDialog().getOpenFileName(plot_handler.plot_window,
                                                 'Open CIF file', '/home',
                                                 'Files (*.cif)')
        cif_path = str(cif_path[0]) if isinstance(cif_path, tuple) else str(cif_path)

        if not cif_path:
            plot_handler.plot_window.uncheck_action_by_text(plot_handler.plot_window.menu_bragg_peaks, "CIF file")
            return

        key = path.basename(cif_path).rsplit('.')[0]
        plot_handler._cif_file = key
        plot_handler._cif_path = cif_path
    else:
        key = plot_handler._cif_file
        cif_path = None
    if key:
        recoil = False
        toggle_overplot_line(plot_handler, plotter_presenter, key, recoil,
                             checked, cif_file=cif_path)


def remove_line(line):
    plt.gca().lines.remove(line)


def plot_overplot_line(x, y, key, recoil, cache):
    color = OVERPLOT_COLORS[key] if key in OVERPLOT_COLORS else 'c'
    if recoil:
        return overplot_line(x, y, color, get_recoil_label(key), cache.rotated)
    else:
        return overplot_line(x, y, color, key, cache.rotated)


def overplot_line(x, y, color, label, rotated):
    if rotated:
        return plt.gca().plot(y, x, color=color, label=label, alpha=.7,
                              picker=PICKER_TOL_PTS)[0]
    else:
        return plt.gca().plot(x, y, color=color, label=label, alpha=.7,
                              picker=PICKER_TOL_PTS)[0]
