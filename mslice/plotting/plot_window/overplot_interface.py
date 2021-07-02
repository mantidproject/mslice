from qtpy.QtWidgets import QFileDialog

import os.path as path


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
    plot_handler.manager.report_as_current()

    if checked:
        plotter_presenter.add_overplot_line(plot_handler.ws_name, key, recoil, cif_file)
    else:
        plotter_presenter.hide_overplot_line(plot_handler.ws_name, key)

    plot_handler.update_legend()
    plot_handler._canvas.draw()

    # Reset current active figure
    if last_active_figure_number is not None:
        plot_handler.manager._current_figs.set_figure_as_current(last_active_figure_number)


def cif_file_powder_line(plot_handler, plotter_presenter, checked):
    if checked:
        cif_path = QFileDialog().getOpenFileName(plot_handler.plot_window,
                                                 'Open CIF file', '/home',
                                                 'Files (*.cif)')
        cif_path = str(cif_path[0]) if isinstance(cif_path, tuple) else str(cif_path)
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
