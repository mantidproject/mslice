from mantid.simpleapi import ConvertToMD, SliceMD
from models.projection.powder.projection_calculator import ProjectionCalculator
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider




class MantidProjectionCalculator(ProjectionCalculator):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def calculate_projection(self, input_workspace, axis1, axis2):
        if axis1 == '|Q|' and axis2 == 'Energy':
            output_workspace = input_workspace + '_QE'
            ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions='|Q|',
                        PreprocDetectorsWS='-')

        elif axis1 == 'Energy' and axis2 == '|Q|':
            output_workspace = input_workspace + '_EQ'
            ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions='|Q|',
                        PreprocDetectorsWS='-')
            output_workspace_handle = self._workspace_provider.get_workspace_handle(output_workspace)
            # Now swapping dim0 and dim1
            dim0 = output_workspace_handle.getDimension(1)
            dim1 = output_workspace_handle.getDimension(0)
            # format into dimension string as expected
            dim0 = dim0.getName() + ',' + str(dim0.getMinimum()) + ',' +\
                   str(dim0.getMaximum()) + ',' + str(dim0.getNBins())
            dim1 = dim1.getName() + ',' + str(dim1.getMinimum()) + ',' +\
                   str(dim1.getMaximum()) + ',' + str(dim1.getNBins())
            SliceMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, AlignedDim0=dim0, AlignedDim1=dim1)

        else:
            raise NotImplementedError("Not implemented axis1 = %s and axis2 = %s" % (axis1, axis2))

