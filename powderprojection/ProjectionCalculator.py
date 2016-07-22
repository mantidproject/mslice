import abc

class ProjectionCalculator(object):
    @abc.abstractmethod
    def calculate_projections(self, input_workspace, output_workspace, qbinning, axis1, axis2):
        pass
    def calculate_suggested_binning(self, input_workspace):
        pass
