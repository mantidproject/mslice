from mantid.simpleapi import ConvertToMD
from models.projection.powder.projection_calculator import ProjectionCalculator

MD_SUFFIX = '_MD'


class MantidProjectionCalculator(ProjectionCalculator):
    def calculate_projection(self, input_workspace, output_workspace, qbinning, axis1, axis2):
        if axis1 == '|Q|' and axis2 == 'Energy':
            ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=input_workspace + MD_SUFFIX, QDimensions='|Q|',
                        PreprocDetectorsWS='-')
        else:
            raise NotImplementedError('MSlice currently only supports projection to Energy vs |Q|')

