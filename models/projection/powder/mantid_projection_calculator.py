
from mantid.simpleapi import AnalysisDataService,ConvertToMD

from models.projection.powder.projection_calculator import ProjectionCalculator
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider


class MantidProjectionCalculator(ProjectionCalculator):
    def calculate_projection(self, input_workspace, output_workspace, axis1, axis2):
        import random,string
        letters = list(string.ascii_lowercase)
        random.shuffle(letters)
        input_workspace = MantidWorkspaceProvider.get_workspace_handle(input_workspace)
        x = ConvertToMD(input_workspace,'|Q|','Direct')
        raise NotImplementedError('Implement me')