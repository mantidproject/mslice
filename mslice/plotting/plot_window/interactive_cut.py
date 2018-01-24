
import matplotlib.patches as patches

VERTICAL = 0
HORIZONTAL = 1
INIT_WIDTH = 0.05

class InteractiveCut(object):

    def __init__(self, slice_plot, canvas, start_pos, end_pos):
        self.slice_plot = slice_plot
        self._canvas = canvas
        self.orient = None
        self.box_width = 0
        self.create_box(start_pos, end_pos)


    def create_box(self, start_pos, end_pos):
        x, y, length = self.calc_box_size(start_pos, end_pos)
        print(x, y, length)
        rect = patches.Rectangle((x,y), length, self.box_width, linewidth=2, edgecolor='r', facecolor=None)
        self._canvas.figure.gca().add_patch(rect)
        self._canvas.draw()

    def calc_box_size(self, start_pos, end_pos):
        x_diff = abs(start_pos[0] - end_pos[0])
        y_diff = abs(start_pos[1] - end_pos[1])
        self.orient = HORIZONTAL if x_diff < y_diff else VERTICAL
        if self.orient == HORIZONTAL:
            self.box_width = INIT_WIDTH * self.slice_plot.y_range[1]
            x = min(start_pos[0], end_pos[0])
            y = (start_pos[1] + end_pos[1] + self.box_width) / 2
            length = max(start_pos[0], end_pos[0]) - x
        else:
            self.box_width = INIT_WIDTH * self.slice_plot.x_range[1]
            y = min(start_pos[1], end_pos[1])
            x = (start_pos[0] + end_pos[0] + self.box_width) / 2
            length = max(start_pos[1], end_pos[1]) - y
        return x, y, length

    def display_coords(self, data_coord):
        return self._canvas.figure.gca().transData.transform(data_coord)