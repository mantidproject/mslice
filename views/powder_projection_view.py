


class PowderView(object):
    def __init__(self):
        raise Exception("This abstract class should not be instantiated")

    def populate_powder_u1(self, u1_options):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_powder_u2(self, u2_options):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_powder_projection_axis(self, axis_options):
        self.populate_powder_u1(axis_options)
        self.populate_powder_u2(axis_options[::-1])

    def get_output_workspace_name(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_powder_projection_units(self, powder_projection_units):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_powder_u1(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_powder_u2(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_powder_units(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")





































