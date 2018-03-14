from matplotlib.widgets import RectangleSelector

from mslice.presenters.slice_plotter_presenter import Axis
from mslice.models.cut.mantid_cut_algorithm import MantidCutAlgorithm


class InteractiveCut(object):

    def __init__(self, slice_plot, canvas, ws_title):
        from mslice.models.cut.matplotlib_cut_plotter import MatplotlibCutPlotter
        self.slice_plot = slice_plot
        self._canvas = canvas
        self._ws_title = ws_title
        self.horizontal = None
        self.connect_event = [None, None, None]
        self._cut_algorithm = MantidCutAlgorithm()
        self._cut_plotter = MatplotlibCutPlotter(self._cut_algorithm)
        self.rect = RectangleSelector(self._canvas.figure.gca(), self.plot_from_mouse_event,
                                      drawtype='box', useblit=True,
                                      button=[1,3], spancoords='pixels', interactive=True)
        self._canvas.draw()

    def plot_from_mouse_event(self, eclick, erelease):
        self.horizontal = abs(erelease.x - eclick.x) > abs(erelease.y - eclick.y)
        self.plot_cut(eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata)
        self._cut_plotter.set_icut(self)
        self.connect_event[2] = self._canvas.mpl_connect('button_press_event', self.clicked)

    def plot_cut(self, x1, x2, y1, y2):
        if x2 > x1 and y2 > y1:
            ax, integration_start, integration_end = self.get_cut_parameters((x1, y1), (x2, y2))
            integration_axis = Axis('', integration_start, integration_end, 0)
            self._cut_plotter.plot_cut(str(self._ws_title), ax, integration_axis, False, None, None, False)

    def get_cut_parameters(self, pos1, pos2):
        start = pos1[not self.horizontal]
        end = pos2[not self.horizontal]
        # hard code step for now. When sliceMD is fixed, can get minimum step with cut_algorithm.get_axis_range()
        step = 0.02
        units = self._canvas.figure.gca().get_xaxis().units if self.horizontal else \
            self._canvas.figure.gca().get_yaxis().units
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

    def save_cut(self):
        x1, x2, y1, y2 = self.rect.extents
        ax, integration_start, integration_end = self.get_cut_parameters((x1, y1), (x2, y2))
        self._cut_plotter.save_cut((str(self._ws_title), ax, integration_start, integration_end, False))
        self.update_workspaces()

    def update_workspaces(self):
        self.slice_plot.update_workspaces()

    def set_workspace_provider(self, workspace_provider):
        self._cut_algorithm.set_workspace_provider(workspace_provider)

    def clear(self):
        self._cut_plotter.set_icut(None)
        self.rect.set_active(False)
        for event in self.connect_event:
            self._canvas.mpl_disconnect(event)
        self._canvas.draw()
