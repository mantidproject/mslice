from matplotlib.widgets import RectangleSelector

from mslice.models.axis import Axis
from mslice.models.cut.cut_functions import output_workspace_name
from mslice.models.workspacemanager.workspace_algorithms import (get_limits)
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter


class InteractiveCut(object):

    def __init__(self, slice_plot, canvas, ws_title):
        self.slice_plot = slice_plot
        self._canvas = canvas
        self._ws_title = ws_title
        self.horizontal = None
        self.connect_event = [None, None, None, None]
        self._cut_plotter_presenter = CutPlotterPresenter()
        self._rect_pos_cache = [0, 0, 0, 0, 0, 0]
        self.rect = RectangleSelector(self._canvas.figure.gca(), self.plot_from_mouse_event,
                                      drawtype='box', useblit=True,
                                      button=[1, 3], spancoords='pixels', interactive=True)

        self.connect_event[3] = self._canvas.mpl_connect('draw_event', self.redraw_rectangle)
        self._canvas.draw()

    def plot_from_mouse_event(self, eclick, erelease):
        # Make axis orientation sticky, until user selects entirely new rectangle.
        rect_pos = [eclick.x, eclick.y, erelease.x, erelease.y,
                    abs(erelease.x - eclick.x), abs(erelease.y - eclick.y)]
        rectangle_changed = all([abs(rect_pos[i] - self._rect_pos_cache[i]) > 0.1 for i in range(6)])
        if rectangle_changed:
            self.horizontal = abs(erelease.x - eclick.x) > abs(erelease.y - eclick.y)
        self.plot_cut(eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
        self._cut_plotter_presenter.store_icut(self._ws_title, self)
        self.connect_event[2] = self._canvas.mpl_connect('button_press_event', self.clicked)
        self._rect_pos_cache = rect_pos

    def plot_cut(self, x1, x2, y1, y2, store=False):
        if x2 > x1 and y2 > y1:
            ax, integration_start, integration_end = self.get_cut_parameters((x1, y1), (x2, y2))
            units = self._canvas.figure.gca().get_yaxis().units if self.horizontal else \
                self._canvas.figure.gca().get_xaxis().units
            integration_axis = Axis(units, integration_start, integration_end, 0)
            self._cut_plotter_presenter.plot_interactive_cut(str(self._ws_title), ax, integration_axis, store)

    def get_cut_parameters(self, pos1, pos2):
        start = pos1[not self.horizontal]
        end = pos2[not self.horizontal]
        units = self._canvas.figure.gca().get_xaxis().units if self.horizontal else \
            self._canvas.figure.gca().get_yaxis().units
        step = get_limits(get_workspace_handle(self._ws_title), units)[2]
        ax = Axis(units, start, end, step)
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
        self._cut_plotter_presenter.set_is_icut(self._ws_title, False)
        self.rect.set_active(False)
        for event in self.connect_event:
            self._canvas.mpl_disconnect(event)
        self._canvas.draw()

    def flip_axis(self):
        self.horizontal = not self.horizontal
        self.plot_cut(*self.rect.extents)

    def window_closing(self):
        self.slice_plot.plot_window.action_interactive_cuts.setChecked(False)
        self.slice_plot.toggle_icut()
