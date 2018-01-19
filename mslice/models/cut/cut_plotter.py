class CutPlotter(object):
    def __init__(self, _cut_algorithm):
        raise Exception('This class is an interface')

    def plot_cut(self, selected_workspace, cut_axis, integration_start, integration_end, norm_to_one, intensity_start,
                 intensity_end, plot_over):
        raise NotImplementedError('This class is an abstract interface')

    def plot_cut_from_x_y_e(self, x, y, e, x_units, selected_workspace, plot_over):
        raise NotImplementedError('This class is an abstract interface')
