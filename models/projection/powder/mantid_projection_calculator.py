from mantid.simpleapi import ConvertToMD
from models.projection.powder.projection_calculator import ProjectionCalculator
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider


MD_SUFFIX = '_QE'


class MantidProjectionCalculator(ProjectionCalculator):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def calculate_projection(self, input_workspace, axis1, axis2):
        output_workspace = input_workspace + MD_SUFFIX
        emode = self._workspace_provider.get_workspace_handle(input_workspace).getEMode().name
        if axis1 == '|Q|' and axis2 == 'Energy':
            ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions='|Q|',
                        PreprocDetectorsWS='-', dEAnalysisMode=emode)
        else:
            raise NotImplementedError('MSlice currently only supports projection to Energy vs |Q|')

