from matplotlib.widgets import RectangleSelector

from mslice.models.axis import Axis
from mslice.models.cut.cut import Cut
from mslice.models.cut.cut_functions import output_workspace_name
from mslice.models.units import EnergyUnits
from mslice.models.workspacemanager.workspace_algorithms import (get_limits)
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.plotting.pyplot import GlobalFigureManager


class InteractiveCut(object):

    def __init__(self, slice_plot, canvas, ws_title):
        self.slice_plot = slice_plot
        self._canvas = canvas
        self._ws_title = ws_title
        self._en_unit = slice_plot.get_slice_cache().energy_axis.e_unit
        self._en_from_meV = EnergyUnits(self._en_unit).factor_from_meV()

        self.horizontal = None
        self.connect_event = [None, None, None, None]
        # We need to access the CutPlotterPresenter instance of the particular CutPlot (window) we are using
        # But there is no way to get without changing the active category then calling the GlobalFigureManager.
        # So we create a new temporary here. After the first time we plot a 1D plot, the correct category is set
        # and we can get the correct CutPlot instance and its CutPlotterPresenter
        self._cut_plotter_presenter = CutPlotterPresenter()
        self._is_initial_cut_plotter_presenter = True
        self._rect_pos_cache = [0, 0, 0, 0, 0, 0]
        self.rect = RectangleSelector(self._canvas.figure.gca(), self.plot_from_mouse_event,
                                      drawtype='box', useblit=True,
                                      button=[1, 3], spancoords='pixels', interactive=True)

        self.connect_event[3] = self._canvas.mpl_connect('draw_event', self.redraw_rectangle)
        self._canvas.draw()
        self.slice_plot.set_cross_cursor()

    def plot_from_mouse_event(self, eclick, erelease):
        # Make axis orientation sticky, until user selects entirely new rectangle.
        rect_pos = [eclick.x, eclick.y, erelease.x, erelease.y,
                    abs(erelease.x - eclick.x), abs(erelease.y - eclick.y)]
        rectangle_changed = all([abs(rect_pos[i] - self._rect_pos_cache[i]) > 0.1 for i in range(6)])
        if rectangle_changed:
            self.horizontal = abs(erelease.x - eclick.x) > abs(erelease.y - eclick.y)
        self.plot_cut(eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
        self.connect_event[2] = self._canvas.mpl_connect('button_press_event', self.clicked)
        self._rect_pos_cache = rect_pos

    def plot_cut(self, x1, x2, y1, y2, store=False):
        if x2 > x1 and y2 > y1:
            ax, integration_start, integration_end = self.get_cut_parameters((x1, y1), (x2, y2))
            units = self._canvas.figure.gca().get_yaxis().units if self.horizontal else \
                self._canvas.figure.gca().get_xaxis().units
            integration_axis = Axis(units, integration_start, integration_end, 0, self._en_unit)
            workspace = get_workspace_handle(self._ws_title)
            cut = Cut(ax, integration_axis, None, None, sample_temp=self.slice_plot.temp, e_fixed=workspace.e_fixed)
            cut.parent_ws_name = self._ws_title
            intensity_method = "scattering_function" if not self.slice_plot.intensity_method else self.slice_plot.intensity_method[5:]
            self._cut_plotter_presenter.plot_interactive_cut(workspace, cut, store, intensity_method)
            self._cut_plotter_presenter.set_is_icut(True)
            if self._is_initial_cut_plotter_presenter:
                # First time we've plotted a 1D cut - get the true CutPlotterPresenter
                self._cut_plotter_presenter = GlobalFigureManager.get_active_figure().plot_handler._cut_plotter_presenter
                self._is_initial_cut_plotter_presenter = False
                GlobalFigureManager.disable_make_current()
                self.slice_plot.plot_window.action_save_cut.setVisible(True)
            self._cut_plotter_presenter.store_icut(self)

    def get_cut_parameters(self, pos1, pos2):
        start = pos1[not self.horizontal]
        end = pos2[not self.horizontal]
        units = self._canvas.figure.gca().get_xaxis().units if self.horizontal else \
            self._canvas.figure.gca().get_yaxis().units
        step = get_limits(get_workspace_handle(self._ws_title), units)[2] * self._en_from_meV
        ax = Axis(units, start, end, step, self._en_unit)
        integration_start = pos1[self.horizontal]
        integration_end = pos2[self.horizontal]
        return ax, integration_start, integration_end

    def clicked(self, event):
        self.connect_event[0] = self._canvas.mpl_connect('motion_notify_event',
                                                         lambda x: self.plot_cut(*self.rect.extents))
        self.connect_event[1] = self._canvas.mpl_connect('button_release_event', self.end_drag)

    def end_drag(self, event):
        self._canvas.mpl_disconnect(self.connect_event[0])
        self._canvas.mpl_disconnect(self.connect_event[1])

    def redraw_rectangle(self, event):
        if self.rect.active:
            self.rect.update()

    def save_cut(self):
        x1, x2, y1, y2 = self.rect.extents
        self.plot_cut(x1, x2, y1, y2, store=True)
        self.update_workspaces()
        ax, integration_start, integration_end = self.get_cut_parameters((x1, y1), (x2, y2))
        return output_workspace_name(str(self._ws_title), integration_start, integration_end)

    def update_workspaces(self):
        self.slice_plot.update_workspaces()

    def clear(self):
        self._cut_plotter_presenter.set_is_icut(False)
        self.rect.set_active(False)
        self.rect.set_visible(False)
        for event in self.connect_event:
            self._canvas.mpl_disconnect(event)

        self._canvas.draw()

    def flip_axis(self):
        self.horizontal = not self.horizontal
        self.plot_cut(*self.rect.extents)

    def window_closing(self):
        self.slice_plot.toggle_interactive_cuts()
        self.slice_plot.plot_window.action_interactive_cuts.setChecked(False)

    def refresh_rect_selector(self, ax):
        extents = self.rect.extents
        self.rect = RectangleSelector(ax, self.plot_from_mouse_event,
                                      drawtype='box', useblit=True,
                                      button=[1, 3], spancoords='pixels', interactive=True)
        self.rect.to_draw.set_visible(True)
        self.rect.extents = extents
        self.slice_plot.set_cross_cursor()

    def store_icut_cut_upon_toggle_and_reset(self):
        self._cut_plotter_presenter.store_icut_cut()
        self._cut_plotter_presenter.set_icut_cut(None)

    def set_icut_intensity_category(self, intensity_method):
        self._cut_plotter_presenter.set_icut_intensity_category(intensity_method)

    def refresh_current_cut(self):
        x1, x2, y1, y2 = self.rect.extents
        self.plot_cut(x1, x2, y1, y2)