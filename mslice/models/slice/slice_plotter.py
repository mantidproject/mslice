
class SlicePlotter():
    def plot_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                   colourmap):
        pass

    def get_available_colormaps(self):
        pass

    def get_available_axis(self, workspace):
        pass

    def get_axis_range(self, workspace, dimension_name):
        pass

    def set_workspace_provider(self, workspace_provider):
        pass

    def sample_temperature(self, ws_name):
        pass

    def show_scattering_function(self, workspace):
        pass

    def show_dynamical_susceptibility(self, workspace):
        pass

    def show_dynamical_susceptibility_magnetic(self, workspace):
        pass

    def add_sample_temperature_field(self, field_name):
        pass

    def update_sample_temperature(self, workspace):
        pass

    def is_sliceable(self, workspace):
        pass
