
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
        self.orient = None
        self.rect = None
        self._cut_algorithm = MantidCutAlgorithm()
        self._cut_plotter = MatplotlibCutPlotter(self._cut_algorithm)
        self.create_cut(start_pos, end_pos)

    def create_cut(self, start_pos, end_pos):
        coords = self.draw_box(start_pos, end_pos)
        # assuming horizontal for now
        x_start = coords[0][0]
        x_end = coords[1][0]
        step = 0.02 # hardcode for now, possibly get default value?
        ax = Axis('MomentumTransfer', x_start, x_end, step)
        integration_start = coords[0][1]
        integration_end = coords[1][1]
        self._cut_plotter.plot_cut(str(self.slice_plot._ws_title), ax, integration_start, integration_end,
                                   False, None, None, False)

    def draw_box(self, start_pos, end_pos):
        x, y, width, height = self.calc_box_size(start_pos, end_pos)
        self.rect = patches.Rectangle((x,y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        self._canvas.figure.gca().add_patch(self.rect)
        self._canvas.draw()
        return self.rect.get_bbox().get_points()

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

    # def display_coords(self, data_coord):
    #     return self._canvas.figure.gca().transData.transform(data_coord)

    def clear(self):
        self.rect.remove()
        del self