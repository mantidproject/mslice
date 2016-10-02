import tempfile
from mantid.simpleapi import ConvertToMD, SliceMD, ConvertSpectrumAxis, PreprocessDetectorsToMD, DeleteWorkspace
from models.projection.powder.projection_calculator import ProjectionCalculator
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider

# unit labels
MOD_Q_LABEL = '|Q|'
THETA_LABEL = '2Theta'
DELTA_E_LABEL = 'DeltaE'

class MantidProjectionCalculator(ProjectionCalculator):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def available_units(self):
        return [MOD_Q_LABEL, THETA_LABEL, DELTA_E_LABEL]

    def _flip_axes(self, output_workspace):
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

    def _getDetWS(self, input_workspace):
        wsdet = next(tempfile._get_candidate_names())
        PreprocessDetectorsToMD(InputWorkspace=input_workspace, OutputWorkspace=wsdet)
        return wsdet

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
            return self._flip_axes(output_workspace)
        elif axis1 == THETA_LABEL and axis2 == DELTA_E_LABEL:
            output_workspace = input_workspace + '_ThE'
            ConvertSpectrumAxis(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, Target='Theta')
            # Work-around for a bug in ConvertToMD.
            wsdet = self._getDetWS(input_workspace) if emode == 'Indirect' else '-'
            ConvertToMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, QDimensions='CopyToMD',
                        PreprocDetectorsWS=wsdet, dEAnalysisMode=emode)
            if emode == 'Indirect':
                DeleteWorkspace(wsdet)
            return self._flip_axes(output_workspace)
        elif axis1 == DELTA_E_LABEL and axis2 == THETA_LABEL:
            output_workspace = input_workspace + '_ETh'
            ConvertSpectrumAxis(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, Target='Theta')
            return ConvertToMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, QDimensions='CopyToMD',
                               PreprocDetectorsWS='-', dEAnalysisMode=emode)
        else:
            raise NotImplementedError("Not implemented axis1 = %s and axis2 = %s" % (axis1, axis2))
