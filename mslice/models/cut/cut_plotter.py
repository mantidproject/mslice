class CutPlotter(object):
    def __init__(self, _cut_algorithm):
        raise Exception('This class is an interface')

    def plot_cut(self, selected_workspace, cut_axis, integration_axis, norm_to_one, intensity_start,
                 intensity_end, plot_over):
        raise NotImplementedError('This class is an abstract interface')

    def save_cut(self, params):
        raise NotImplementedError('This class is an abstract interface')

    def set_icut(self, icut):
        raise NotImplementedError('This class is an abstract interface')

    def plot_cut_from_xye(self, x, y, e, x_units, selected_workspace, out_ws_name, plot_over):
        raise NotImplementedError('This class is an abstract interface')
