from mantid.simpleapi import ConvertToMD
from models.projection.powder.projection_calculator import ProjectionCalculator

MD_SUFFIX = '_MD'


class MantidProjectionCalculator(ProjectionCalculator):
    def calculate_projection(self, input_workspace, axis1, axis2):
        output_workspace = input_workspace + MD_SUFFIX
        if axis1 == '|Q|' and axis2 == 'Energy':
            ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions='|Q|',
                        PreprocDetectorsWS='-')
        else:
            raise NotImplementedError('MSlice currently only supports projection to Energy vs |Q|')

