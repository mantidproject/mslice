
import matplotlib.patches as patches

from mslice.presenters.slice_plotter_presenter import Axis
from mslice.models.cut.mantid_cut_algorithm import MantidCutAlgorithm

VERTICAL = 0
HORIZONTAL = 1
INIT_WIDTH = 0.05

class InteractiveCut(object):

    def __init__(self, slice_plot, canvas, start_pos, end_pos):
        from mslice.models.cut.matplotlib_cut_plotter import MatplotlibCutPlotter
        self.slice_plot = slice_plot
        self._canvas = canvas
        self.background = self._canvas.copy_from_bbox(self._canvas.figure.gca().bbox)
        self.orient = None
        self.rect = None
        self.coords = None
        self.dragging = False
        self._cut_algorithm = MantidCutAlgorithm()
        self._cut_plotter = MatplotlibCutPlotter(self._cut_algorithm)
        self.create_box(start_pos, end_pos)
        self.create_cut(False)
        self._canvas.mpl_connect('button_press_event', self.clicked)

    def create_cut(self, update):
        # assuming horizontal for now
        x_start = self.coords[0][0]
        x_end = self.coords[1][0]
        step = 0.02 # hardcode for now, possibly get default value?
        ax = Axis('MomentumTransfer', x_start, x_end, step)
        integration_start = self.coords[0][1]
        integration_end = self.coords[1][1]
        if update:
            self._cut_plotter.update_cut(str(self.slice_plot._ws_title), ax, integration_start, integration_end,
                                       False, None, None, False)
        else:
            self._cut_plotter.plot_cut(str(self.slice_plot._ws_title), ax, integration_start, integration_end,
                                   False, None, None, False)

    def create_box(self, start_pos, end_pos):
        x, y, width, height = self.calc_box_size(start_pos, end_pos)
        self.draw_box(x, y, width, height)
        self._canvas.draw()

    def update_coords(self, delta_x, delta_y):
        x = self.coords[0][0] + delta_x
        y = self.coords[0][1] + delta_y
        width = self.rect.get_width()
        height = self.rect.get_height()
        self.rect.remove()
        self.draw_box(x, y, width, height)

    def draw_box(self, x, y, width, height):
        self.rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        self._canvas.restore_region(self.background)
        self._canvas.figure.gca().add_patch(self.rect)
        self._canvas.figure.gca().draw_artist(self.rect)
        self._canvas.blit(self._canvas.figure.gca().bbox)
        self.coords = self.rect.get_bbox().get_points()

    def calc_box_size(self, start_pos, end_pos):
        x_diff = abs(start_pos[0] - end_pos[0])
        y_diff = abs(start_pos[1] - end_pos[1])
        self.orient = HORIZONTAL if x_diff > y_diff else VERTICAL
        print(self.orient)
        if self.orient == HORIZONTAL:
            height = INIT_WIDTH * self.slice_plot.y_range[1]
            x = min(start_pos[0], end_pos[0])
            y = (start_pos[1] + end_pos[1] - height) / 2
            width = max(start_pos[0], end_pos[0]) - x
            return x, y, width, height
        else:
            width = INIT_WIDTH * self.slice_plot.x_range[1]
            y = min(start_pos[1], end_pos[1])
            x = (start_pos[0] + end_pos[0] - width) / 2
            height = max(start_pos[1], end_pos[1]) - y
        return x, y, width, height

    def inside_cut(self, xpos, ypos):
        return self.coords[0][0] < xpos < self.coords[1][0] and self.coords[0][1] < ypos < self.coords[1][1]

    def clicked(self, event):
        if self.dragging:
            self._canvas.mpl_disconnect(self.connect_event)
            self.dragging = False
        else:
            self.dragging = True
            if self.inside_cut(event.xdata, event.ydata):
                self.drag_orig_pos = [event.xdata, event.ydata]
                self.connect_event = self._canvas.mpl_connect('motion_notify_event', self.drag)

    def drag(self, event):
        xchange = event.xdata - self.drag_orig_pos[0]
        ychange = event.ydata - self.drag_orig_pos[1]
        self.drag_orig_pos[0] = event.xdata
        self.drag_orig_pos[1] = event.ydata
        self.update_coords(xchange, ychange)
        self.create_cut(True)

    def clear(self):
        self.rect.remove()
        del self

    def none(self, event):
        pass