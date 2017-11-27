
class CutAlgorithm(object):
    def compute_cut(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        pass

    def compute_cut_xye(self, selected_workspace, cut_axis, integration_start, integration_end, is_norm):
        pass

    def is_cuttable(self, workspace):
        pass

    def is_cut(self, workspace):
        pass

    def get_cut_params(self, cut_workspace):
        pass

    def get_available_axis(self, workspace):
        pass

    def get_other_axis(self, workspace, axis):
        pass

    def get_axis_range(self, workspace, dimension_name):
        pass

    def set_workspace_provider(self, workspace_provider):
        pass

    def set_saved_cut_parameters(self, workspace, axis, parameters):
        pass

    def get_saved_cut_parameters(self, workspace, axis=None):
        pass

    def is_axis_saved(self, workspace, axis):
        pass
