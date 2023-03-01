from __future__ import (absolute_import, division, print_function)
import uuid
from mantid.api import PythonAlgorithm, MatrixWorkspaceProperty, IMDEventWorkspaceProperty
from mantid.kernel import FloatArrayProperty, Direction, StringMandatoryValidator
from mantid.simpleapi import (DeleteWorkspace, SliceMD, PreprocessDetectorsToMD, ConvertToMD, SofQW3, ConvertSpectrumAxis)
from ...labels import DELTA_E_LABEL, MOD_Q_LABEL, THETA_LABEL

class MakeProjection(PythonAlgorithm):

    def PyInit(self):
        self.declareProperty(MatrixWorkspaceProperty('InputWorkspace', "", direction=Direction.Input))
        self.declareProperty('Axis1', "", StringMandatoryValidator())
        self.declareProperty('Axis2', "", StringMandatoryValidator())
        self.declareProperty(FloatArrayProperty('Limits', direction=Direction.Input))
        self.declareProperty('EMode', "Direct", StringMandatoryValidator())
        self.declareProperty('ProjectionType', "QE", StringMandatoryValidator(), doc='Q or Theta projection')
        self.declareProperty(IMDEventWorkspaceProperty('OutputWorkspace', "", direction=Direction.Output))

    def PyExec(self):
        """Calculate the projection workspace AND return a python handle to it"""
        input_workspace  = self.getProperty('InputWorkspace').value
        axis1 = self.getProperty('Axis1').value
        axis2 = self.getProperty('Axis2').value
        emode = self.getProperty('EMode').value
        projection_type = self.getProperty('ProjectionType').value

        if projection_type == 'QE':
            new_ws = self._calcQEproj(input_workspace, emode, axis1, axis2)
        else:
            new_ws = self._calcThetaEproj(input_workspace, emode, axis1, axis2)
        self.setProperty('OutputWorkspace', new_ws)


    def category(self):
        return 'MSlice'

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
        return SliceMD(outputWorkspace=output_workspace, StoreInADS=False, InputWorkspace=output_workspace,
                       AlignedDim0=dim0, AlignedDim1=dim1)

    def _getDetWS(self, input_workspace):
        """ Precalculates the detector workspace for ConvertToMD - workaround for bug for indirect geometry """
        wsdet = str(uuid.uuid4().hex)
        wsdet = PreprocessDetectorsToMD(OutputWorkspace=wsdet, StoreInADS=False, InputWorkspace=input_workspace)
        return wsdet

    def _calcQEproj(self, input_workspace, emode, axis1, axis2):
        """ Carries out either the Q-E or E-Q projections """
        # For indirect geometry and large datafiles (likely to be using a 1-to-1 mapping use ConvertToMD('|Q|')
        numSpectra = input_workspace.getNumberHistograms()
        if emode == 'Indirect' or numSpectra > 1000:
            retval = ConvertToMD(InputWorkspace=input_workspace, QDimensions=MOD_Q_LABEL,
                                 PreprocDetectorsWS='-', dEAnalysisMode=emode, StoreInADS=False)
            if axis1 == DELTA_E_LABEL and axis2 == MOD_Q_LABEL:
                retval = self._flip_axes(retval)
        # Otherwise first run SofQW3 to rebin it in |Q| properly before calling ConvertToMD with CopyToMD
        else:
            limits = self.getProperty('Limits').value
            limits = ','.join([str(limits[i]) for i in [0, 2, 1]])
            retval = SofQW3(InputWorkspace=input_workspace, QAxisBinning=limits, Emode=emode, StoreInADS=False)
            retval = ConvertToMD(InputWorkspace=retval, QDimensions='CopyToMD', PreprocDetectorsWS='-',
                                 dEAnalysisMode=emode, StoreInADS=False)
            if axis1 == MOD_Q_LABEL:
                retval = self._flip_axes(retval)
        return retval

    def _calcThetaEproj(self, input_workspace, emode, axis1, axis2):
        """ Carries out either the 2Theta-E or E-2Theta projections """
        retval = ConvertSpectrumAxis(InputWorkspace=input_workspace, Target='Theta')
        # Work-around for a bug in ConvertToMD.
        wsdet = self._getDetWS(input_workspace) if emode == 'Indirect' else '-'
        retval = ConvertToMD(InputWorkspace=retval, QDimensions='CopyToMD', PreprocDetectorsWS=wsdet,
                             dEAnalysisMode=emode, StoreInADS=False)
        if emode == 'Indirect':
            DeleteWorkspace(wsdet)
        if axis1 == THETA_LABEL:
            retval = self._flip_axes(retval)
        return retval
