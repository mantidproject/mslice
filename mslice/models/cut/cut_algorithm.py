
class CutAlgorithm(object):
    def compute_cut(self, selected_workspace, cut_axis, integration_axis, is_norm):
        pass

    def compute_cut_xye(self, selected_workspace, cut_axis, integration_axis, is_norm):
        pass

    def get_arrays_from_workspace(self, workspace):
        pass

    def is_cuttable(self, workspace):
        pass

    def is_cut(self, workspace):
        pass

    def get_available_axis(self, workspace):
        pass

    def get_other_axis(self, workspace, axis):
        pass

    def get_axis_range(self, workspace, dimension_name):
        pass

    def set_saved_cut_parameters(self, workspace, axis, parameters):
        pass

    def get_saved_cut_parameters(self, workspace, axis=None):
        pass

    def is_axis_saved(self, workspace, axis):
        pass
