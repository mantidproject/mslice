

class CutAlgorithm(object):
    def compute_cut(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        pass

    def compute_cut_xye(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        pass

    def is_slice(self, workspace):
        pass

    def is_cut(self, workspace):
        pass

    def get_cut_params(self, cut_workspace):
        pass

    def get_available_axis(self, workspace):