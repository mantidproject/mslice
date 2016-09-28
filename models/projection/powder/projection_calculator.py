import abc

class ProjectionCalculator(object):
    @abc.abstractmethod
    def available_units(self):
        pass

    @abc.abstractmethod
    def calculate_projection(self, input_workspace, axis1, axis2):
        pass

    def calculate_suggested_binning(self, input_workspace):
        pass
