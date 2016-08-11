#import abc


class PowderView(object):

    #@abc.abstractmethod
    def populate_powder_u1(self, u1_options):
        pass

    #@abc.abstractmethod
    def populate_powder_u2(self, u2_options):
        pass

    def populate_powder_projection_axis(self, axis_options):
        self.populate_powder_u1(axis_options)
        self.get_powder_u2()

    #@abc.abstractmethod
    def get_output_workspace_name(self):
        pass

    #@abc.abstractmethod
    def populate_powder_projection_units(self, powder_projection_units):
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





































