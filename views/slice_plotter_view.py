

class SlicePlotterView:
    def __init__(self):
        raise Exception("This abstract class must not be instantiated")

    def get_slice_x_axis(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_x_start(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_x_end(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_x_step(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_y_start(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_y_axis(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_y_end(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_y_step(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_intensity_start(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_intensity_end(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_is_norm_to_one(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_smoothing(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_slice_colourmap(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_x_params(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_y_params(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_intensity_params(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_smoothing_params(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_x_units(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_y_units(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_select_one_workspace(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_plot_parameters(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_colormap_options(self,colormaps):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_slice_x_options(self, options):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_slice_y_options(self, options):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_slice_x_params(self, x_start, x_end, x_step):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_slice_y_params(self, y_start, y_end, y_step):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def clear_input_fields(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def clear_displayed_error(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")


