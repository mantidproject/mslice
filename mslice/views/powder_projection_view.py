


class PowderView(object):
    error_occurred = None

    def __init__(self):
        raise Exception("This abstract class should not be instantiated")

    def populate_powder_u1(self, u1_options):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def populate_powder_u2(self, u2_options):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

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

    def set_powder_u1(self, name):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def set_powder_u2(self, name):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def clear_displayed_error(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_presenter(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")
