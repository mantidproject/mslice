from functools import partial

from qtpy import QtWidgets
from qtpy.QtCore import Qt

import matplotlib.colors as colors
from matplotlib.legend import Legend
from matplotlib.text import Text

from mslice.models.colors import to_hex, name_to_color
from mslice.models.units import get_sample_temperature_from_string
from mslice.presenters.plot_options_presenter import SlicePlotOptionsPresenter
from mslice.presenters.quick_options_presenter import quick_options, check_latex
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.plotting.plot_window.cachable_input_dialog import QCacheableInputDialog
from mslice.plotting.plot_window.iplot import IPlot
from mslice.plotting.plot_window.interactive_cut import InteractiveCut
from mslice.plotting.plot_window.plot_options import SlicePlotOptions
from mslice.plotting.plot_window.overplot_interface import (
    _update_overplot_lines,
    _update_powder_lines,
    toggle_overplot_line,
    cif_file_powder_line,
)
from mslice.plotting.pyplot import GlobalFigureManager
from mslice.scripting import generate_script
from mslice.util.compat import legend_set_draggable
from mslice.util.intensity_correction import IntensityType, IntensityCache
from mslice.models.intensity_correction_algs import sample_temperature

from typing import Callable

DEFAULT_LABEL_SIZE = 10
DEFAULT_TITLE_SIZE = 12
DEFAULT_FONT_SIZE_STEP = 1


class SlicePlot(IPlot):
    def __init__(self, figure_manager, slice_plotter_presenter, workspace_name):
        self.manager = figure_manager
        self.plot_window = figure_manager.window
        self._canvas = self.plot_window.canvas
        self._slice_plotter_presenter = slice_plotter_presenter
        self.ws_name = workspace_name
        self._arb_nuclei_rmm = None
        self._cif_file = None
        self._cif_path = None
        self._legends_shown = True
        self._legend_dict = {}

        # Interactive cuts
        self.icut = None
        self.icut_event = [None, None]

        self.setup_connections(self.plot_window)

        self.intensity = False
        self.intensity_type = IntensityType.SCATTERING_FUNCTION
        self.temp_dependent = False
        self.temp = None
        self.default_options = None
        self.plot_fonts_properties = [
            "title_size",
            "x_range_font_size",
            "y_range_font_size",
            "x_label_size",
            "y_label_size",
            "colorbar_label_size",
            "colorbar_range_font_size",
        ]
        self.plot_window.set_manual_temp_log_enabled(False)

    def save_default_options(self):
        self.default_options = {
            "colorbar_label": self.colorbar_label,
            "colorbar_label_size": DEFAULT_LABEL_SIZE,
            "colorbar_log": self.colorbar_log,
            "colorbar_range": self.colorbar_range,
            "colorbar_range_font_size": DEFAULT_LABEL_SIZE,
            "intensity": self.intensity,
            "intensity_type": self.intensity_type,
            "temp": self.temp,
            "temp_dependent": self.temp_dependent,
            "title": self.ws_name,
            "title_size": DEFAULT_TITLE_SIZE,
            "x_label": r"$|Q|$ ($\mathrm{\AA}^{-1}$)",
            "x_label_size": DEFAULT_LABEL_SIZE,
            "x_grid": False,
            "x_range": self.x_range,
            "x_range_font_size": DEFAULT_LABEL_SIZE,
            "y_label": "Energy Transfer (meV)",
            "y_label_size": DEFAULT_LABEL_SIZE,
            "y_grid": False,
            "y_range": self.y_range,
            "y_range_font_size": DEFAULT_LABEL_SIZE,
            "legend": True,
        }

    def setup_connections(self, plot_window):
        plot_window.redraw.connect(self._canvas.draw)
        plot_window.action_gen_script.setVisible(True)
        plot_window.action_gen_script_clipboard.setVisible(True)
        plot_window.menu_information.setDisabled(False)
        plot_window.menu_intensity.setDisabled(False)
        plot_window.action_toggle_legends.setVisible(True)
        plot_window.action_keep.setVisible(True)
        plot_window.action_make_current.setVisible(True)
        plot_window.action_save_image.setVisible(True)
        plot_window.action_plot_options.setVisible(True)
        plot_window.action_interactive_cuts.setVisible(True)
        plot_window.action_interactive_cuts.triggered.connect(
            self.toggle_interactive_cuts
        )
        plot_window.action_save_cut.setVisible(False)
        plot_window.action_save_cut.triggered.connect(self.save_icut)
        plot_window.action_flip_axis.setVisible(False)
        plot_window.action_flip_axis.triggered.connect(self.flip_icut)
        plot_window.action_waterfall.setVisible(False)

        plot_window.action_sqe.triggered.connect(
            partial(
                self.show_intensity_plot,
                plot_window.action_sqe,
                self._slice_plotter_presenter.show_scattering_function,
                False,
            )
        )
        plot_window.action_chi_qe.triggered.connect(
            partial(
                self.show_intensity_plot,
                plot_window.action_chi_qe,
                self._slice_plotter_presenter.show_dynamical_susceptibility,
                True,
            )
        )
        plot_window.action_chi_qe_magnetic.triggered.connect(
            partial(
                self.show_intensity_plot,
                plot_window.action_chi_qe_magnetic,
                self._slice_plotter_presenter.show_dynamical_susceptibility_magnetic,
                True,
            )
        )
        plot_window.action_d2sig_dw_de.triggered.connect(
            partial(
                self.show_intensity_plot,
                plot_window.action_d2sig_dw_de,
                self._slice_plotter_presenter.show_d2sigma,
                False,
            )
        )
        plot_window.action_symmetrised_sqe.triggered.connect(
            partial(
                self.show_intensity_plot,
                plot_window.action_symmetrised_sqe,
                self._slice_plotter_presenter.show_symmetrised,
                True,
            )
        )
        plot_window.action_gdos.triggered.connect(
            partial(
                self.show_intensity_plot,
                plot_window.action_gdos,
                self._slice_plotter_presenter.show_gdos,
                True,
            )
        )
        plot_window.action_set_temp_log.triggered.connect(
            self._get_prev_and_set_sample_temperature
        )

        plot_window.action_hydrogen.triggered.connect(
            partial(toggle_overplot_line, self, self._slice_plotter_presenter, 1, True)
        )
        plot_window.action_deuterium.triggered.connect(
            partial(toggle_overplot_line, self, self._slice_plotter_presenter, 2, True)
        )
        plot_window.action_helium.triggered.connect(
            partial(toggle_overplot_line, self, self._slice_plotter_presenter, 4, True)
        )
        plot_window.action_arbitrary_nuclei.triggered.connect(
            self.arbitrary_recoil_line
        )
        plot_window.action_aluminium.triggered.connect(
            partial(
                toggle_overplot_line,
                self,
                self._slice_plotter_presenter,
                "Aluminium",
                False,
            )
        )
        plot_window.action_copper.triggered.connect(
            partial(
                toggle_overplot_line,
                self,
                self._slice_plotter_presenter,
                "Copper",
                False,
            )
        )
        plot_window.action_niobium.triggered.connect(
            partial(
                toggle_overplot_line,
                self,
                self._slice_plotter_presenter,
                "Niobium",
                False,
            )
        )
        plot_window.action_tantalum.triggered.connect(
            partial(
                toggle_overplot_line,
                self,
                self._slice_plotter_presenter,
                "Tantalum",
                False,
            )
        )
        plot_window.action_cif_file.triggered.connect(
            partial(cif_file_powder_line, self, self._slice_plotter_presenter)
        )
        plot_window.action_gen_script.triggered.connect(self.generate_script)
        plot_window.action_gen_script_clipboard.triggered.connect(
            lambda: self.generate_script(clipboard=True)
        )

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
        plot_window.action_set_temp_log.triggered.disconnect()
        plot_window.action_hydrogen.triggered.disconnect()
        plot_window.action_deuterium.triggered.disconnect()
        plot_window.action_helium.triggered.disconnect()
        plot_window.action_arbitrary_nuclei.triggered.disconnect()
        plot_window.action_aluminium.triggered.disconnect()
        plot_window.action_copper.triggered.disconnect()
        plot_window.action_niobium.triggered.disconnect()
        plot_window.action_tantalum.triggered.disconnect()
        plot_window.action_cif_file.triggered.disconnect()
        plot_window.action_gen_script.triggered.disconnect()

    def window_closing(self):
        if self.icut is not None:
            self.icut.clear()
            self.icut.window_closing()
            self.icut = None

    def plot_options(self):
        SlicePlotOptionsPresenter(
            SlicePlotOptions(self.plot_window, redraw_signal=self.plot_window.redraw),
            self,
        )

    def plot_clicked(self, x, y):
        bounds = self.calc_figure_boundaries()
        if bounds["x_label"] < y < bounds["title"]:
            if bounds["y_label"] < x < bounds["colorbar_label"]:
                if y < bounds["x_range"]:
                    quick_options(
                        "x_range", self, redraw_signal=self.plot_window.redraw
                    )
                elif x < bounds["y_range"]:
                    quick_options(
                        "y_range", self, redraw_signal=self.plot_window.redraw
                    )
                elif x > bounds["colorbar_range"]:
                    quick_options(
                        "colorbar_range",
                        self,
                        self.colorbar_log,
                        redraw_signal=self.plot_window.redraw,
                    )

    def object_clicked(self, target):
        if isinstance(target, Legend):
            return
        elif isinstance(target, Text):
            quick_options(target, self, redraw_signal=self.plot_window.redraw)
        else:
            quick_options(target, self)
            self.update_legend()
            self._canvas.draw()

    def update_legend(self):
        axes = self._canvas.figure.gca()

        if self._legends_shown:
            self._add_or_renew_legend(axes)

        if self._canvas.manager.plot_handler.icut is not None:
            self._canvas.manager.plot_handler.icut.rect.ax = axes

    def _add_or_renew_legend(self, axes):
        handles, labels = axes.get_legend_handles_labels()

        if handles:
            # Uses the 'upper right' location because 'best' causes very slow plotting for large datasets.
            axes.legend(handles, labels, fontsize="medium", loc="upper right")
            legend_set_draggable(axes.get_legend(), True)
        else:
            legend = axes.get_legend()
            if legend:
                legend.remove()

    def change_axis_scale(self, colorbar_range, logarithmic):
        current_axis = self._canvas.figure.gca()
        colormesh = current_axis.collections[0]
        vmin, vmax = colorbar_range

        if logarithmic:
            if vmin <= float(0):
                vmin = 0.001
            if vmax <= float(0):
                vmax = 0.001

            norm = colors.LogNorm(vmin, vmax)
        else:
            norm = colors.Normalize(vmin, vmax)

        label = self.colorbar_label
        colormesh.colorbar.remove()
        colormesh.set_clim((vmin, vmax))
        colormesh.set_norm(norm)
        self._canvas.figure.colorbar(colormesh)
        self.colorbar_label = label

    def get_line_options(self, target):
        line_options = {
            "label": target.get_label(),
            "legend": None,
            "shown": None,
            "color": to_hex(target.get_color()),
            "style": target.get_linestyle(),
            "width": str(target.get_linewidth()),
            "marker": target.get_marker(),
            "error_bar": None,
        }
        return line_options

    def set_line_options(self, line, line_options):
        line.set_label(line_options["label"])
        line.set_linestyle(line_options["style"])
        line.set_marker(line_options["marker"])
        line.set_color(name_to_color(line_options["color"]))
        line.set_linewidth(line_options["width"])

    def calc_figure_boundaries(self):
        fig_x, fig_y = self._canvas.figure.get_size_inches() * self._canvas.figure.dpi
        bounds = {}
        bounds["y_label"] = fig_x * 0.07
        bounds["y_range"] = fig_x * 0.12
        bounds["colorbar_range"] = fig_x * 0.75
        bounds["colorbar_label"] = fig_x * 0.86
        bounds["title"] = fig_y * 0.9
        bounds["x_range"] = fig_y * 0.09
        bounds["x_label"] = fig_y * 0.05
        return bounds

    def arbitrary_recoil_line(self):
        recoil = True
        checked = self.plot_window.action_arbitrary_nuclei.isChecked()
        if checked:
            self._arb_nuclei_rmm, confirm = QtWidgets.QInputDialog.getInt(
                self.plot_window, "Arbitrary Nuclei", "Enter relative mass:", min=1
            )
            if confirm:
                toggle_overplot_line(
                    self,
                    self._slice_plotter_presenter,
                    self._arb_nuclei_rmm,
                    recoil,
                    checked,
                )
            else:
                self.plot_window.action_arbitrary_nuclei.setChecked(not checked)
        else:
            toggle_overplot_line(
                self,
                self._slice_plotter_presenter,
                self._arb_nuclei_rmm,
                recoil,
                checked,
            )

    def _reset_intensity(self):
        options = self.plot_window.menu_intensity.actions()
        for op in options:
            op.setChecked(False)

    def selected_intensity(self):
        options = self.plot_window.menu_intensity.actions()
        for option in options:
            if option.isChecked():
                return option

    def set_intensity(self, intensity):
        self._reset_intensity()
        intensity.setChecked(True)

    def show_intensity_plot(self, action, slice_plotter_method, temp_dependent):
        last_active_figure_number, disable_make_current_after_plot = (
            self.manager.report_as_current_and_return_previous_status()
        )
        if not self.default_options:
            self.save_default_options()
        self.default_options["temp_dependent"] = temp_dependent
        self.temp_dependent = temp_dependent
        self.default_options["intensity"] = True
        self.intensity = True
        self.default_options["intensity_type"] = (
            IntensityCache.get_intensity_type_from_desc(
                slice_plotter_method.__name__[5:]
            )
        )
        self.intensity_type = self.default_options["intensity_type"]

        if action.isChecked():
            previous = self.selected_intensity()
            self.set_intensity(action)
            cbar_log = self.colorbar_log
            cbar_range = self.colorbar_range
            title = self.title
            if temp_dependent:
                self.plot_window.set_manual_temp_log_enabled(True)
                if not self._run_temp_dependent(slice_plotter_method, previous):
                    self.manager.reset_current_figure_as_previous(
                        last_active_figure_number, disable_make_current_after_plot
                    )
                    return
            else:
                self.plot_window.set_manual_temp_log_enabled(False)
                slice_plotter_method(self.ws_name)
            self.update_canvas(cbar_range, cbar_log, title)
        else:
            action.setChecked(True)
        self.manager.reset_current_figure_as_previous(
            last_active_figure_number, disable_make_current_after_plot
        )
        if self.icut:
            self.icut.refresh_current_cut()

    def update_canvas(self, cbar_range, cbar_log, title):
        self.change_axis_scale(cbar_range, cbar_log)
        self.title = title
        self.manager.update_grid()
        self._update_lines()
        self._canvas.draw()

    def _run_temp_dependent(
        self, slice_plotter_method: Callable, previous: QtWidgets.QAction
    ) -> bool:
        try:
            slice_plotter_method(self.ws_name)
        except (
            ValueError
        ):  # sample temperature not yet set, get it and reattempt method
            # First, try to get it from the temperature cache:
            cached_temp_pack = self._slice_plotter_presenter.get_cached_sample_temp()
            if cached_temp_pack is not None:
                cached_temp_log, is_field = cached_temp_pack
            else:
                cached_temp_log = None
                is_field = None
            if cached_temp_log is not None:
                self._handle_temperature_input(cached_temp_log, is_field, False)
                return True
            if self._set_sample_temperature(previous):
                slice_plotter_method(self.ws_name)
            else:  # failed to get sample temperature
                return False
        return True

    def _get_prev_and_set_sample_temperature(self) -> bool:
        """
        Helper for the sake of simplifying the call for changing the temp via the menu.
        """
        return self._set_sample_temperature(self.selected_intensity())

    def _set_sample_temperature(self, previous: QtWidgets.QAction) -> bool:
        try:
            temp_value_raw, field, is_cached = self.ask_sample_temperature_field(
                str(self.ws_name)
            )
            temperature_found = self._handle_temperature_input(
                temp_value_raw, field, is_cached
            )
        except RuntimeError:  # if cancel is clicked, go back to previous selection
            temperature_found = False
        if not temperature_found:
            self.set_intensity(previous)
        return temperature_found

    def _handle_temperature_input(
        self, temp_value_raw: str, field: bool, is_cached: bool
    ) -> bool:
        if field:
            temp_value = sample_temperature(self.ws_name, [temp_value_raw])
        else:
            temp_value = get_sample_temperature_from_string(temp_value_raw)

        if temp_value is None or temp_value < 0:
            self.plot_window.display_error(
                "Invalid value entered for sample temperature. Enter a value in Kelvin \
                                            or a sample log field."
            )
            return False

        self.default_options["temp"] = temp_value
        self.temp = temp_value
        self._slice_plotter_presenter.set_sample_temperature(
            self.ws_name, temp_value, temp_value_raw, is_cached, field
        )
        return True

    def ask_sample_temperature_field(self, ws_name: str) -> tuple[str, bool, bool]:
        text = "Sample temperature not found. Select the sample temperature field or enter a value in Kelvin:"
        ws = get_workspace_handle(ws_name)
        try:
            keys = ws.raw_ws.run().keys()
        except AttributeError:
            keys = ws.raw_ws.getExperimentInfo(0).run().keys()
        temp_field, is_cached, confirm = QCacheableInputDialog.ask_for_input(
            self.plot_window, "Sample Temperature", text, keys
        )
        if not confirm:
            raise RuntimeError("sample_temperature_dialog cancelled")
        else:
            return str(temp_field), temp_field in keys, is_cached

    def _update_recoil_lines(self):
        """Updates the recoil overplots lines when intensity type changes"""
        lines = {
            self.plot_window.action_hydrogen: [1, True, ""],
            self.plot_window.action_deuterium: [2, True, ""],
            self.plot_window.action_helium: [4, True, ""],
            self.plot_window.action_arbitrary_nuclei: [self._arb_nuclei_rmm, True, ""],
        }
        _update_overplot_lines(self._slice_plotter_presenter, self.ws_name, lines)

    def _update_lines(self):
        """Updates the powder/recoil overplots lines when intensity type changes"""
        self._update_recoil_lines()
        _update_powder_lines(self, self._slice_plotter_presenter)
        self.update_legend()
        self._canvas.draw()

    def toggle_interactive_cuts(self):
        self.toggle_icut_button()
        self.toggle_icut()

    def toggle_icut_button(self):
        if not self.icut:
            self.manager.picking_connected(False)
            if self.plot_window.action_zoom_in.isChecked():
                self.plot_window.action_zoom_in.setChecked(False)
                self.plot_window.action_zoom_in.triggered.emit(False)  # turn off zoom
            self.plot_window.action_zoom_in.setEnabled(False)
            self.plot_window.action_keep.trigger()
            self.plot_window.action_keep.setEnabled(False)
            self.plot_window.action_make_current.setEnabled(False)
            self.plot_window.action_flip_axis.setVisible(True)
            self.plot_window.menu_intensity.setDisabled(True)
        else:
            self.manager.picking_connected(True)
            self.plot_window.action_zoom_in.setEnabled(True)
            self.plot_window.action_keep.setEnabled(True)
            self.plot_window.action_make_current.setEnabled(True)
            self.plot_window.action_save_cut.setVisible(False)
            self.plot_window.action_flip_axis.setVisible(False)
            self._canvas.setCursor(Qt.ArrowCursor)
            self.icut.set_icut_intensity_category(self.intensity_type)
            self.icut.store_icut_cut_upon_toggle_and_reset(False)
            self.plot_window.menu_intensity.setDisabled(False)

    def toggle_icut(self):
        if self.icut is not None:
            self.icut.clear()
            self.icut = None
            GlobalFigureManager.enable_make_current()
        else:
            self.icut = InteractiveCut(self, self._canvas, self.ws_name)

    def save_icut(self):
        self.icut.save_cut()

    def flip_icut(self):
        self.icut.flip_axis()

    def get_slice_cache(self):
        return self._slice_plotter_presenter.get_slice_cache(self.ws_name)

    def get_cached_workspace(self):
        cached_slice = self.get_slice_cache()
        return getattr(cached_slice, self.intensity_type.name.lower())

    def update_workspaces(self):
        self._slice_plotter_presenter.update_displayed_workspaces()

    def on_newplot(self):
        # This callback should be activated by a call to pcolormesh
        self.plot_window.action_hydrogen.setChecked(False)
        self.plot_window.action_deuterium.setChecked(False)
        self.plot_window.action_helium.setChecked(False)
        self.plot_window.action_arbitrary_nuclei.setChecked(False)
        self.plot_window.action_aluminium.setChecked(False)
        self.plot_window.action_copper.setChecked(False)
        self.plot_window.action_niobium.setChecked(False)
        self.plot_window.action_tantalum.setChecked(False)
        self.plot_window.action_cif_file.setChecked(False)

    def generate_script(self, clipboard=False):
        try:
            generate_script(self.ws_name, None, self, self.plot_window, clipboard)
        except Exception as e:
            self.plot_window.display_error(e.message)

    @property
    def colorbar_label(self):
        return self._canvas.figure.get_axes()[1].get_ylabel()

    @colorbar_label.setter
    def colorbar_label(self, value):
        self._canvas.figure.get_axes()[1].set_ylabel(
            value, labelpad=20, rotation=270, picker=5
        )

    @property
    def colorbar_label_size(self):
        return self._canvas.figure.get_axes()[1].yaxis.label.get_size()

    @colorbar_label_size.setter
    def colorbar_label_size(self, value):
        self._canvas.figure.get_axes()[1].yaxis.label.set_size(value)

    @property
    def colorbar_range(self):
        return self._canvas.figure.gca().collections[0].get_clim()

    @colorbar_range.setter
    def colorbar_range(self, value):
        self.change_axis_scale(value, self.colorbar_log)

    @property
    def colorbar_range_font_size(self):
        return self._canvas.figure.get_axes()[1].get_yticklabels()[0].get_size()

    @colorbar_range_font_size.setter
    def colorbar_range_font_size(self, value):
        self._canvas.figure.get_axes()[1].tick_params(
            axis="y", which="both", labelsize=value
        )

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
        if check_latex(value):
            self.manager.title = value
        else:
            self.plot_window.display_error("invalid latex string")

    @property
    def title_size(self):
        return self.manager.title_size

    @title_size.setter
    def title_size(self, value):
        self.manager.title_size = value

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
    def x_label_size(self):
        return self.manager.x_label_size

    @x_label_size.setter
    def x_label_size(self, value):
        self.manager.x_label_size = value

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
    def y_label_size(self):
        return self.manager.y_label_size

    @y_label_size.setter
    def y_label_size(self, value):
        self.manager.y_label_size = value

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
    def show_legends(self):
        return self._legends_shown

    @show_legends.setter
    def show_legends(self, value):
        self._legends_shown = value

    def is_changed(self, item):
        if self.default_options is None:
            return False
        return self.default_options[item] != getattr(self, item)

    @property
    def y_log(self):  # needed for interface consistency with cut plot
        return False

    @staticmethod
    def _get_overplot_datum():  # needed for interface consistency with cut plot
        return 0

    def set_cross_cursor(self):
        self._canvas.setCursor(Qt.CrossCursor)

    @property
    def all_fonts_size(self):
        font_sizes_config = {}
        for p in self.plot_fonts_properties:
            font_sizes_config[p] = getattr(self, p)

        return font_sizes_config

    @all_fonts_size.setter
    def all_fonts_size(self, values: dict):
        for key in values:
            setattr(self, key, values[key])

    def increase_all_fonts(self):
        for p in self.plot_fonts_properties:
            setattr(self, p, getattr(self, p) + DEFAULT_FONT_SIZE_STEP)

    def decrease_all_fonts(self):
        for p in self.plot_fonts_properties:
            setattr(self, p, getattr(self, p) - DEFAULT_FONT_SIZE_STEP)
