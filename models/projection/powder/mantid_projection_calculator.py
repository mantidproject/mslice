from mantid.simpleapi import ConvertToMD, SliceMD, TransformMD
from models.projection.powder.projection_calculator import ProjectionCalculator
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider

# unit labels
MOD_Q_LABEL = '|Q|'
DELTA_E_LABEL = 'DeltaE'
MEV_LABEL = 'meV'
WAVENUMBER_LABEL = 'cm-1'

class MantidProjectionCalculator(ProjectionCalculator):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def available_axes(self):
        return [MOD_Q_LABEL, DELTA_E_LABEL]

    def available_units(self):
        return [WAVENUMBER_LABEL, MEV_LABEL]

    def calculate_projection(self, input_workspace, axis1, axis2, units):
        """Calculate the projection workspace AND return a python handle to it"""
        emode = self._workspace_provider.get_workspace_handle(input_workspace).getEMode().name
        if axis1 == MOD_Q_LABEL and axis2 == DELTA_E_LABEL:
            scale = [1, 8.06554]
            output_workspace = input_workspace + '_QE' + ('_cm' if units == WAVENUMBER_LABEL else '')
            ConvertToMD(InputWorkspace=input_workspace, OutputWorkspace=output_workspace, QDimensions=MOD_Q_LABEL,
                        PreprocDetectorsWS='-', dEAnalysisMode=emode)
        elif axis1 == DELTA_E_LABEL and axis2 == MOD_Q_LABEL:
            scale = [8.06554, 1]
            output_workspace = input_workspace + '_EQ' + ('_cm' if units == WAVENUMBER_LABEL else '')
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
            SliceMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, AlignedDim0=dim0,
                    AlignedDim1=dim1)
        else:
            raise NotImplementedError("Not implemented axis1 = %s and axis2 = %s" % (axis1, axis2))
        # Now scale the energy axis if required - ConvertToMD always gives DeltaE in meV
        if units == WAVENUMBER_LABEL:
            TransformMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, Scaling=scale)
            self._workspace_provider.get_workspace_handle(output_workspace).setComment('MSlice_in_wavenumber')
        elif units != MEV_LABEL:
            raise NotImplementedError("Unit %s not recognised. Only 'meV' and 'cm-1' implemented." % (units))
