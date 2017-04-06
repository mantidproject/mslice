import abc

class ProjectionCalculator(object):
    @abc.abstractmethod
    def available_axes(self):
        pass

    @abc.abstractmethod
    def available_units(self):
        pass

    @abc.abstractmethod
    def calculate_projection(self, input_workspace, axis1, axis2, units):
        pass

    def set_workspace_provider(self, workspace_provider):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")
