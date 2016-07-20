#import abc

class MainView(object):
    #__metaclass__ = abc.ABCMeta
    #TODO metaclass?

    #@abc.abstractmethod
    def update_log(self, log):
        pass

    # <These Might go>
    #@abc.abstractmethod
    def set_memory_usage(self, memory_usage):
        pass

    #@abc.abstractmethod
    def set_CPU_usage(self, cpu_usage):
        pass
    # </These Might go>

    #@abc.abstractmethod
    def set_application_progress(self, progress):
        pass

    #@abc.abstractmethod
    def populate_powder_u1(self, u1_options):
        pass

    #@abc.abstractmethod
    def populate_powder_u2(self, u2_options):
        pass

    #@abc.abstractmethod
    def populate_powder_projection_axis(self, axis_options):
        """Populate both u1 and u2 with the options from the list of strings axis_options"""
        pass

    #@abc.abstractmethod
    def populate_powder_projection_units(self, powder_projection_units):
        pass

    #TODO Implement Single Crystal Functionality

    #@abc.abstractmethod
    def populate_slice_x(self, x_axis_options):
        pass

    #@abc.abstractmethod
    def populate_slice_y(self, y_axis_options):
        pass

    #@abc.abstractmethod
    def populate_slice_axis(self, axis_options):
        """Populate the options for x and y axis with options from list of strings axis_options"""
        pass

    #@abc.abstractmethod
    def populate_slice_colourmaps(self, colourmaps):
        pass

    #@abc.abstractmethod
    def populate_cut_axis(self, cut_axis_options):
        pass


    #@abc.abstractmethod
    def get_powder_u1(self):
        pass

    #@abc.abstractmethod
    def get_powder_u2(self):
        pass

    #@abc.abstractmethod
    def get_powder_units(self):
        pass

    #TODO Implement Single Crystal Functionality


    #@abc.abstractmethod
    def get_slice_x_axis(self):
        pass

    #@abc.abstractmethod
    def get_slice_x_start(self):
        pass

    #@abc.abstractmethod
    def get_slice_x_end(self):
        pass
    #@abc.abstractmethod
    def get_slice_x_step(self):
        pass

    #@abc.abstractmethod
    def get_slice_y_start(self):
        pass

    #@abc.abstractmethod
    def get_slice_y_axis(self):
        pass

    #@abc.abstractmethod
    def get_slice_y_end(self):
        pass

    #@abc.abstractmethod
    def get_slice_y_step(self):
        pass

    #@abc.abstractmethod
    def get_intensity_start(self):
        pass

    #@abc.abstractmethod
    def get_intensity_end(self):
        pass

    #@abc.abstractmethod
    def get_slice_is_norm_to_one(self):
        pass

    #@abc.abstractmethod
    def get_slice_smoothing(self):
        pass

    #@abc.abstractmethod
    def get_slice_colourmap(self):
        pass

    #@abc.abstractmethod
    def get_cut_axis(self):
        pass

    #@abc.abstractmethod
    def get_cut_start(self):
        pass

    #@abc.abstractmethod
    def get_cut_end(self):
        pass

    #@abc.abstractmethod
    def get_cut_step(self):
        pass

    #@abc.abstractmethod
    def get_cut_integration_start(self):
        pass

    #@abc.abstractmethod
    def get_cut_integration_end(self):
        pass

    #@abc.abstractmethod
    def get_cut_integration_width(self):
        pass

    #@abc.abstractmethod
    def get_cut_intensity_start(self):
        pass

    #@abc.abstractmethod
    def get_cut_intensity_end(self):
        pass

    #@abc.abstractmethod
    def get_cut_is_norm_to_one(self):
        pass

    #@abc.abstractmethod
    def get_cut_smoothing(self):
        pass

    #@abc.abstractmethod
    def error_select_only_one_workspace(self):
        pass










































