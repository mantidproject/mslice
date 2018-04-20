from __future__ import (absolute_import, division, print_function)
import uuid
from mantid.simpleapi import ConvertToMD, SliceMD, TransformMD, ConvertSpectrumAxis, PreprocessDetectorsToMD
from mantid.simpleapi import RenameWorkspace, DeleteWorkspace, SofQW3
from mslice.models.workspacemanager.mantid_workspace_provider import get_workspace_handle, propagate_properties
from mslice.models.projection.powder.projection_calculator import ProjectionCalculator

# unit labels
MOD_Q_LABEL = '|Q|'
THETA_LABEL = '2Theta'
DELTA_E_LABEL = 'DeltaE'
MEV_LABEL = 'meV'
WAVENUMBER_LABEL = 'cm-1'

class MantidProjectionCalculator(ProjectionCalculator):

    def available_axes(self):
        return [MOD_Q_LABEL, THETA_LABEL, DELTA_E_LABEL]

    def available_units(self):
        return [MEV_LABEL, WAVENUMBER_LABEL]

    def _flip_axes(self, output_workspace):
        """ Transposes the x- and y-axes """
        # Now swapping dim0 and dim1
        dim0 = output_workspace.getDimension(1)
        dim1 = output_workspace.getDimension(0)
        # format into dimension string as expected
        dim0 = dim0.getName() + ',' + str(dim0.getMinimum()) + ',' +\
            str(dim0.getMaximum()) + ',' + str(dim0.getNBins())
        dim1 = dim1.getName() + ',' + str(dim1.getMinimum()) + ',' +\
            str(dim1.getMaximum()) + ',' + str(dim1.getNBins())
        return SliceMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, AlignedDim0=dim0,
                       AlignedDim1=dim1)

    def _getDetWS(self, input_workspace):
        """ Precalculates the detector workspace for ConvertToMD - workaround for bug for indirect geometry """
        wsdet = str(uuid.uuid4().hex)
        PreprocessDetectorsToMD(InputWorkspace=input_workspace, OutputWorkspace=wsdet)
        return wsdet

    def _calcQEproj(self, input_workspace_name, emode, axis1, axis2):
        """ Carries out either the Q-E or E-Q projections """
        input_workspace = get_workspace_handle(input_workspace_name)
        output_workspace = input_workspace_name + ('_QE' if axis1 == MOD_Q_LABEL else '_EQ')
        # For indirect geometry and large datafiles (likely to be using a 1-to-1 mapping use ConvertToMD('|Q|')
        numSpectra = input_workspace.raw_ws.getNumberHistograms()
        if emode == 'Indirect' or numSpectra > 1000:
            retval = ConvertToMD(InputWorkspace=input_workspace.raw_ws, OutputWorkspace=output_workspace, QDimensions=MOD_Q_LABEL,
                                 PreprocDetectorsWS='-', dEAnalysisMode=emode)
            if axis1 == DELTA_E_LABEL and axis2 == MOD_Q_LABEL:
                retval = self._flip_axes(output_workspace)
        # Otherwise first run SofQW3 to rebin it in |Q| properly before calling ConvertToMD with CopyToMD
        else:
            limits = input_workspace.limits['Momentum Transfer']
            limits = ','.join([str(limits[i]) for i in [0, 2, 1]])
            SofQW3(InputWorkspace=input_workspace.raw_ws, OutputWorkspace=output_workspace, QAxisBinning=limits, Emode=emode)
            retval = ConvertToMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, QDimensions='CopyToMD',
                                 PreprocDetectorsWS='-', dEAnalysisMode=emode)
            if axis1 == MOD_Q_LABEL and axis2 == DELTA_E_LABEL:
                retval = self._flip_axes(retval)
        return retval, output_workspace

    def _calcThetaEproj(self, input_workspace_name, emode, axis1, axis2):
        """ Carries out either the 2Theta-E or E-2Theta projections """
        input_workspace = get_workspace_handle(input_workspace_name)
        output_workspace = input_workspace_name + ('_ThE' if axis1 == THETA_LABEL else '_ETh')
        ConvertSpectrumAxis(InputWorkspace=input_workspace.raw_ws, OutputWorkspace=output_workspace, Target='Theta')
        # Work-around for a bug in ConvertToMD.
        wsdet = self._getDetWS(input_workspace) if emode == 'Indirect' else '-'
        retval = ConvertToMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, QDimensions='CopyToMD',
                             PreprocDetectorsWS=wsdet, dEAnalysisMode=emode)
        if emode == 'Indirect':
            DeleteWorkspace(wsdet)
        if axis1 == THETA_LABEL and axis2 == DELTA_E_LABEL:
            retval = self._flip_axes(output_workspace)
        return retval, output_workspace

    def calculate_projection(self, input_workspace_name, axis1, axis2, units):
        """Calculate the projection workspace AND return a python handle to it"""
        input_workspace = get_workspace_handle(input_workspace_name)
        if not input_workspace.is_PSD:
            raise RuntimeError('Cannot calculate projections for non-PSD workspaces')
        emode = input_workspace.e_mode
        # Calculates the projection - can have Q-E or 2theta-E or their transpose.
        if (axis1 == MOD_Q_LABEL and axis2 == DELTA_E_LABEL) or (axis1 == DELTA_E_LABEL and axis2 == MOD_Q_LABEL):
            new_ws, output_workspace = self._calcQEproj(input_workspace_name, emode, axis1, axis2)
        elif (axis1 == THETA_LABEL and axis2 == DELTA_E_LABEL) or (axis1 == DELTA_E_LABEL and axis2 == THETA_LABEL):
            new_ws, output_workspace = self._calcThetaEproj(input_workspace_name, emode, axis1, axis2)
        else:
            raise NotImplementedError("Not implemented axis1 = %s and axis2 = %s" % (axis1, axis2))
        # Now scale the energy axis if required - ConvertToMD always gives DeltaE in meV
        if units == WAVENUMBER_LABEL:
            scale = [1, 8.06554] if axis2 == DELTA_E_LABEL else [8.06544, 1]
            new_ws = TransformMD(InputWorkspace=output_workspace, OutputWorkspace=output_workspace, Scaling=scale)
            new_ws = RenameWorkspace(InputWorkspace=output_workspace, OutputWorkspace=output_workspace+'_cm')
            new_ws.setComment('MSlice_in_wavenumber')
            output_workspace += '_cm'
        elif units != MEV_LABEL:
            raise NotImplementedError("Unit %s not recognised. Only 'meV' and 'cm-1' implemented." % (units))
        return propagate_properties(input_workspace, new_ws)

    def validate_workspace(self, ws):
        workspace = get_workspace_handle(ws)
        try:
            axes = [workspace.raw_ws.getAxis(0), workspace.raw_ws.getAxis(1)]
            if not all([ax.isSpectra() or ax.getUnit().unitID() == 'DeltaE' for ax in axes]):
                raise AttributeError
        except (AttributeError, IndexError):
            raise TypeError('Input workspace for projection calculation must be a reduced '
                            'data workspace with a spectra and energy transfer axis.')
