import numpy as np

import matplotlib.patches as patches
from matplotlib.lines import Line2D

from mslice.presenters.slice_plotter_presenter import Axis
from mslice.models.cut.mantid_cut_algorithm import MantidCutAlgorithm

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
        self.horizontal = None
        self.rect = None
        self.coords = None
        self.dragging = False
        self.highlight = None
        self.units = None
        self.connect_event = [None, None]

        self._cut_algorithm = MantidCutAlgorithm()
        self._cut_plotter = MatplotlibCutPlotter(self._cut_algorithm)
        self.create_box(start_pos, end_pos)
        self.create_cut(False)
        self._canvas.mpl_connect('button_press_event', self.clicked)
        self.connect_event[0] = self._canvas.mpl_connect('motion_notify_event', self.select_box)

    def create_cut(self, update):
        start, end, step, integration_start, integration_end = self.get_cut_parameters(self.coords, self.horizontal)
        ax = Axis(self.units, start, end, step)
        if update:
            self._cut_plotter.update_cut(str(self.slice_plot._ws_title), ax, integration_start, integration_end,
                                         False, None, None)
        else:
            self._cut_plotter.plot_cut(str(self.slice_plot._ws_title), ax, integration_start, integration_end,
                                       False, None, None, False)

    def get_cut_parameters(self, coords, horizontal):
        start = self.coords[0][not horizontal]
        end = self.coords[1][not horizontal]
        step = 0.02  # hardcode for now, possibly get default value?
        integration_start = self.coords[0][horizontal]
        integration_end = self.coords[1][horizontal]
        return start, end, step, integration_start, integration_end

    def create_box(self, start_pos, end_pos):
        self.set_box_orientation(start_pos, end_pos)
        if self.horizontal:
            height, x, y, width = self.box_dimensions(start_pos, end_pos, self.slice_plot.y_range[1])
        else:
            width, y, x, height = self.box_dimensions(start_pos, end_pos, self.slice_plot.x_range[1])
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

    def set_box_orientation(self, start_pos, end_pos):
        x_diff = abs(start_pos[0] - end_pos[0])
        y_diff = abs(start_pos[1] - end_pos[1])
        self.horizontal =  x_diff > y_diff
        self.units = self._canvas.figure.gca().get_xaxis().units if self.horizontal else \
            self._canvas.figure.gca().get_yaxis().units

    def box_dimensions(self, start_pos, end_pos, axis_maximum):
        """get length, width and co-ords of the bottom left corner of the box. The x or y coordinates of start_pos
        and end_pos are accessed depending on self.horizontal (direction of the cut)."""
        length1 = INIT_WIDTH * axis_maximum
        pos1 = min(start_pos[not self.horizontal], end_pos[not self.horizontal])
        pos2 = (start_pos[self.horizontal] + end_pos[self.horizontal] - length1) / 2
        length2 = max(start_pos[not self.horizontal], end_pos[not self.horizontal]) - pos1
        return length1, pos1, pos2, length2

    def inside_cut(self, xpos, ypos):
        return self.coords[0][0] < xpos < self.coords[1][0] and self.coords[0][1] < ypos < self.coords[1][1]

    def clicked(self, event):
        if self.dragging:
            self.end_transform()
        else:
            self.drag_orig_pos = [event.xdata, event.ydata]
            if self.highlight:
                self.side_extending = self.closest_side(event.x, event.y)
                self.start_transform(self.extend)
            elif self.inside_cut(event.xdata, event.ydata):
                self.start_transform(self.drag)

    def end_transform(self):
        self.dragging = False
        self._canvas.mpl_disconnect(self.connect_event[0])
        self._canvas.mpl_disconnect(self.connect_event[1])

    def start_transform(self, motion_slot):
        self.dragging = True
        self.connect_event[0] = self._canvas.mpl_connect('motion_notify_event', motion_slot)
        self.connect_event[1] = self._canvas.mpl_connect('button_release_event', self.clicked)

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
        """calculates which side of the rectangle is closest to the given coordinates. Parameters must be given
        in terms of the actual location, NOT the data coordinates from the axes."""
        coords = self._canvas.figure.gca().transData.transform(self.coords)
        dist = []
        midpoints = [[], [], [], []]

        # get midpoint of each line
        mid_vertical = (coords[0][1] + coords[1][1]) / 2
        mid_horizontal = (coords[0][0] + coords[1][0]) / 2
        midpoints[LEFT] = [coords[0][0], mid_vertical]
        midpoints[RIGHT] = [coords[1][0], mid_vertical]
        midpoints[TOP] = [mid_horizontal, coords[1][1]]
        midpoints[BOTTOM] = [mid_horizontal, coords[0][1]]

        # use pythagoras to get distance from each midpoint to (x,y)
        for point in midpoints:
            dist.append(np.hypot(x - point[0], y - point[1]))
        return dist

    def select_box(self, event):
        highlight_side = None
        if not self.inside_cut(event.xdata, event.ydata):
            dist = self.dist_to_sides(event.x, event.y)
            if min(dist) <= 50:
                highlight_side = dist.index(min(dist))
        self.highlight_side(highlight_side)

    def highlight_side(self, side):
        line = True
        if side is None:
            line = False
        elif side < 2: # left or right side
            x0 = self.coords[0][0] if side == LEFT else self.coords[1][0]
            x1 = x0
            y0 = self.coords[0][1]
            y1 = self.coords[1][1]
        else: # top or bottom side
            x0 = self.coords[0][0]
            x1 = self.coords[1][0]
            y0 = self.coords[1][1] if side == TOP else self.coords[0][1]
            y1 = y0
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
