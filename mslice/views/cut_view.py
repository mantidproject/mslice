

class CutView:
    error_occurred = None
    busy = None

    def get_cut_axis(self):
        pass

    def get_cut_axis_start(self):
        pass

    def get_cut_axis_end(self):
        pass

    def get_cut_axis_step(self):
        pass

    def get_integration_start(self):
        pass

    def get_integration_end(self):
        pass

    def get_integration_width(self):
        pass

    def get_intensity_start(self):
        pass

    def get_intensity_end(self):
        pass

    def get_intensity_is_norm_to_one(self):
        pass

    def get_smoothing(self):
        pass

    def get_presenter(self):
        pass

    def set_cut_axis(self, axis_name):
        pass

    def set_minimum_step(self, value):
        pass

    def error_select_a_workspace(self):
        pass

    def error_invalid_cut_axis_parameters(self):
        pass

    def error_invalid_integration_parameters(self):
        pass

    def error_invalid_intensity_parameters(self):
        pass

    def error_invalid_width(self):
        pass

    def error_current_selection_invalid(self):
        pass

    def populate_cut_axis_options(self,options):
        pass

    def populate_cut_params(self, cut_start=None, cut_end=None, cut_step=None):
        pass

    def populate_integration_params(self, integration_start=None, integration_end=None):
        pass

    def enable(self):
        pass

    def disable(self):
        pass

    def plotting_params_only(self):
        pass

    def force_normalization(self):
        pass

    def get_input_fields(self):
        pass

    def populate_input_fields(self, saved_input):
        pass

    def clear_input_fields(self, **kwargs):
        pass

    def clear_displayed_error(self):
        pass
