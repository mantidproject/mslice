import abc

class ProjectionCalculator(object):
    @abc.abstractmethod
    def calculate_projection(self, input_workspace, output_workspace, axis1, axis2):
        pass

    def calculate_suggested_binning(self, input_workspace):
        pass
