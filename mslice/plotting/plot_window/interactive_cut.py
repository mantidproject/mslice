import numpy as np

import matplotlib.patches as patches
from matplotlib.lines import Line2D

from mslice.presenters.slice_plotter_presenter import Axis
from mslice.models.cut.mantid_cut_algorithm import MantidCutAlgorithm

VERTICAL = 0
HORIZONTAL = 1
INIT_WIDTH = 0.05
LEFT = 0
RIGHT = 1
TOP = 2
BOTTOM = 3

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
        self.highlight = None
        self._cut_algorithm = MantidCutAlgorithm()
        self._cut_plotter = MatplotlibCutPlotter(self._cut_algorithm)
        self.create_box(start_pos, end_pos)
        self.create_cut(False)
        self.connect_event = [None, None]
        self._canvas.mpl_connect('button_press_event', self.clicked)
        self.connect_event[0] = self._canvas.mpl_connect('motion_notify_event', self.select_box)

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
                                       False, None, None)
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
        if self.highlight:
            self.side_clicked(event)
            return
        if self.dragging:
            self._canvas.mpl_disconnect(self.connect_event[0])
            self._canvas.mpl_disconnect(self.connect_event[1])
            self.dragging = False
        else:
            if self.inside_cut(event.xdata, event.ydata):
                self.dragging = True
                self.drag_orig_pos = [event.xdata, event.ydata]
                self.connect_event[0] = self._canvas.mpl_connect('motion_notify_event', self.drag)
                self.connect_event[1] = self._canvas.mpl_connect('button_release_event', self.clicked)

    def side_clicked(self, event):
        if self.dragging:
            self._canvas.mpl_disconnect(self.connect_event[0])
            self._canvas.mpl_disconnect(self.connect_event[1])
            self.dragging = False
        else:
            self.dragging = True
            x_click = event.xdata
            y_click = event.ydata
            self.drag_orig_pos = [x_click, y_click]
            self.side_extending = self.closest_side(event.x, event.y)
            self.connect_event[0] = self._canvas.mpl_connect('motion_notify_event', self.extend)
            self.connect_event[1] = self._canvas.mpl_connect('button_release_event', self.side_clicked)

    def extend(self, event):
        delta_x = self.drag_orig_pos[0] - event.xdata
        delta_y = self.drag_orig_pos[1] - event.ydata
        x = self.coords[0][0]
        y = self.coords[0][1]
        width = self.rect.get_width()
        height = self.rect.get_height()
        if self.side_extending == RIGHT:
            width -= delta_x
        elif self.side_extending == LEFT:
            x -= delta_x
            width += delta_x
        elif self.side_extending == BOTTOM:
            y -= delta_y
            height += delta_y
        elif self.side_extending == TOP:
            height -= delta_y
        self.rect.remove()
        self.drag_orig_pos[0] = event.xdata
        self.drag_orig_pos[1] = event.ydata
        self.draw_box(x, y, width, height)
        self.create_cut(True)

    def drag(self, event):
        xchange = event.xdata - self.drag_orig_pos[0]
        ychange = event.ydata - self.drag_orig_pos[1]
        self.drag_orig_pos[0] = event.xdata
        self.drag_orig_pos[1] = event.ydata
        self.update_coords(xchange, ychange)
        self.create_cut(True)

    def closest_side(self, x, y):
        dist = self.dist_to_sides(x, y)
        return dist.index(min(dist))

    def dist_to_sides(self, x, y):
        '''calculates which side of the rectangle is closest to the given coordinates. parameters must be given
        in terms of the actual location, NOT the data coordinates from the axes.'''
        coords = self._canvas.figure.gca().transData.transform(self.coords)
        dist = []
        centre_points = [[], [], [], []]
        centre_points[LEFT] = [coords[0][0], (coords[0][1] + coords[1][1]) / 2]
        centre_points[RIGHT] = [coords[1][0], (coords[0][1] + coords[1][1]) / 2]
        centre_points[TOP] = [(coords[0][0] + coords[1][0]) / 2, coords[1][1]]
        centre_points[BOTTOM] = [(coords[0][0] + coords[1][0]) / 2, coords[0][1]]

        dist.append(np.hypot(x - centre_points[0][0], y - centre_points[0][1]))
        dist.append(np.hypot(x - centre_points[1][0], y - centre_points[1][1]))
        dist.append(np.hypot(x - centre_points[2][0], y - centre_points[2][1]))
        dist.append(np.hypot(x - centre_points[3][0], y - centre_points[3][1]))
        return dist

    def select_box(self, event):
        highlight_side = None
        if not self.inside_cut(event.xdata, event.ydata):
            dist = self.dist_to_sides(event.x, event.y)
            if min(dist) <= 50:
                highlight_side = dist.index(min(dist))
        self.highlight_side(highlight_side)

    def highlight_side(self, side):
        line = False
        if side == LEFT:
            x0 = self.coords[0][0]
            x1 = x0
            y0 = self.coords[0][1]
            y1 = self.coords[1][1]
            line = True
        elif side == RIGHT:
            x0 = self.coords[1][0]
            x1 = x0
            y0 = self.coords[0][1]
            y1 = self.coords[1][1]
            line = True
        elif side == TOP:
            x0 = self.coords[0][0]
            x1 = self.coords[1][0]
            y0 = self.coords[1][1]
            y1 = y0
            line = True
        elif side == BOTTOM:
            x0 = self.coords[0][0]
            x1 = self.coords[1][0]
            y0 = self.coords[0][1]
            y1 = y0
            line = True
        if self.highlight is not None:
            self.highlight.set_visible(False)
            self._canvas.figure.gca().lines.remove(self.highlight)
            self.highlight = None
            self._canvas.draw()
        if line:
            self.highlight = Line2D([x0 ,x1],[y0, y1], linewidth=2, color='r')
            self._canvas.figure.gca().add_line(self.highlight)
            self._canvas.figure.gca().draw_artist(self.highlight)
            self._canvas.blit(self._canvas.figure.gca().bbox)

    def clear(self):
        self.rect.remove()
        self._canvas.draw()
