<<<<<<< HEAD
import tempfile
from mantid.simpleapi import ConvertToMD, SliceMD, ConvertSpectrumAxis, PreprocessDetectorsToMD, DeleteWorkspace
=======
from mantid.simpleapi import ConvertToMD, SliceMD, TransformMD
>>>>>>> master
from models.projection.powder.projection_calculator import ProjectionCalculator
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider

# unit labels
MOD_Q_LABEL = '|Q|'
THETA_LABEL = '2Theta'
DELTA_E_LABEL = 'DeltaE'
MEV_LABEL = 'meV'
WAVENUMBER_LABEL = 'cm-1'

class MantidProjectionCalculator(ProjectionCalculator):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def available_units(self):
        return [MOD_Q_LABEL, THETA_LABEL, DELTA_E_LABEL]

    def available_units(self):
        return [WAVENUMBER_LABEL, MEV_LABEL]

    def _flip_axes(self, output_workspace):
        """ Transposes the x- and y-axes """
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
        """ Precalculates the detector workspace for ConvertToMD - workaround for bug for indirect geometry """
        wsdet = next(tempfile._get_candidate_names())
        PreprocessDetectorsToMD(InputWorkspace=input_workspace, OutputWorkspace=wsdet)
        return wsdet

    def _calcQEproj(self, input_workspace, axis1, axis2):
        """ Carries out either the Q-E or E-Q projections """
        output_workspace = input_workspace + ('_QE' if axis1 == MOD_Q_LABEL else '_EQ')
        retval = ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions=MOD_Q_LABEL,
                             PreprocDetectorsWS='-', dEAnalysisMode=emode)
        if axis1 == DELTA_E_LABEL and axis2 == MOD_Q_LABEL:
            retval = self._flip_axes(output_workspace)
        return retval

    def _calcThetaEProj(self, input_workspace, axis1, axis2):
        """ Carries out either the 2Theta-E or E-2Theta projections """
        output_workspace = input_workspace + ('_ThE' if axis1 == THETA_LABEL else '_ETh')
        ConvertSpectrumAxis(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, Target='Theta')
        # Work-around for a bug in ConvertToMD.
        wsdet = self._getDetWS(input_workspace) if emode == 'Indirect' else '-'
        retval = ConvertToMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, QDimensions='CopyToMD',
                             PreprocDetectorsWS=wsdet, dEAnalysisMode=emode)
        if emode == 'Indirect':
            DeleteWorkspace(wsdet)
        if axis1 == THETA_LABEL and axis2 == DELTA_E_LABEL:
            retval = self._flip_axes(output_workspace)
        return retval

    def calculate_projection(self, input_workspace, axis1, axis2, units):
        """Calculate the projection workspace AND return a python handle to it"""
        emode = self._workspace_provider.get_workspace_handle(input_workspace).getEMode().name
        # Calculates the projection - can have Q-E or 2theta-E or their transpose. 
        if (axis1 == MOD_Q_LABEL and axis2 == DELTA_E_LABEL) or (axis1 == DELTA_E_LABEL and axis2 == MOD_Q_LABEL):
            retval = _calcQEproj(input_workspace, axis1, axis2)
        elif (axis1 == THETA_LABEL and axis2 == DELTA_E_LABEL) or (axis1 == DELTA_E_LABEL and axis2 == THETA_LABEL):
            retval = _calcThetaEproj(input_workspace, axis1, axis2)
        else:
            raise NotImplementedError("Not implemented axis1 = %s and axis2 = %s" % (axis1, axis2))
        # Now scale the energy axis if required - ConvertToMD always gives DeltaE in meV
        scale = [1, 8.06554] if axis2 == DELTA_E_LABEL else [8.06544, 1]
        if units == WAVENUMBER_LABEL:
            retval = TransformMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, Scaling=scale)
            retval.setComment('MSlice_in_wavenumber')
        elif units != MEV_LABEL:
            raise NotImplementedError("Unit %s not recognised. Only 'meV' and 'cm-1' implemented." % (units))
        return retval
