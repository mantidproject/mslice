from mantid.simpleapi import ConvertToMD, SliceMD
from models.projection.powder.projection_calculator import ProjectionCalculator
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider

# unit labels
MOD_Q_LABEL = '|Q|'
DELTA_E_LABEL = 'DeltaE'

class MantidProjectionCalculator(ProjectionCalculator):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def available_units(self):
        return [MOD_Q_LABEL, DELTA_E_LABEL]

    def calculate_projection(self, input_workspace, axis1, axis2):
        """Calculate the projection workspace AND return a python handle to it"""
        emode = self._workspace_provider.get_workspace_handle(input_workspace).getEMode().name
        if axis1 == MOD_Q_LABEL and axis2 == DELTA_E_LABEL:
            output_workspace = input_workspace + '_QE'
            return ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions=MOD_Q_LABEL,
                        PreprocDetectorsWS='-', dEAnalysisMode=emode)
        elif axis1 == DELTA_E_LABEL and axis2 == MOD_Q_LABEL:
            output_workspace = input_workspace + '_EQ'
            ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions=MOD_Q_LABEL,
                        PreprocDetectorsWS='-', dEAnalysisMode=emode)
            output_workspace_handle = self._workspace_provider.get_workspace_handle(output_workspace)
            # Now swapping dim0 and dim1
            dim0 = output_workspace_handle.getDimension(1)
            dim1 = output_workspace_handle.getDimension(0)
            # format into dimension string as expected
            dim0 = dim0.getName() + ',' + str(dim0.getMinimum()) + ',' +\
                   str(dim0.getMaximum()) + ',' + str(dim0.getNBins())
            dim1 = dim1.getName() + ',' + str(dim1.getMinimum()) + ',' +\
                   str(dim1.getMaximum()) + ',' + str(dim1.getNBins())
            return SliceMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, AlignedDim0=dim0,
                    AlignedDim1=dim1)
        else:
            raise NotImplementedError("Not implemented axis1 = %s and axis2 = %s" % (axis1, axis2))

