import numpy as np

from mantid.api import PythonAlgorithm, WorkspaceProperty
from mantid.kernel import Direction, PropertyManagerProperty, StringMandatoryValidator, StringListValidator
from mantid.simpleapi import BinMD, ConvertSpectrumAxis, CreateMDHistoWorkspace, Rebin2D, SofQW3, TransformMD, \
    ConvertToMD, DeleteWorkspace, CreateSimulationWorkspace, AddSampleLog, CopyLogs, Integration, Rebin, Transpose, \
    IntegrateMDHistoWorkspace

from mslice.models.alg_workspace_ops import fill_in_missing_input, get_number_of_steps
from mslice.models.axis import Axis
from mslice.models.labels import is_momentum, is_twotheta
from mslice.models.units import EnergyUnits
from mslice.workspace.helperfunctions import attribute_to_log
from .cut_normalisation import normalize_workspace


class Cut(PythonAlgorithm):

    def PyInit(self):
        self.declareProperty(WorkspaceProperty('InputWorkspace', '', direction=Direction.Input))
        self.declareProperty(PropertyManagerProperty('CutAxis', {}, direction=Direction.Input),
                             doc='MSlice Axis object as a dictionary')
        self.declareProperty(PropertyManagerProperty('IntegrationAxis', {}, direction=Direction.Input),
                             doc='MSlice Axis object as a dictionary')
        self.declareProperty('EMode', 'Direct', StringMandatoryValidator())
        self.declareProperty('PSD', False)
        self.declareProperty('NormToOne', False)
        self.declareProperty('IgnorePartialOverlaps', False)
        self.declareProperty('Algorithm', 'Rebin', StringListValidator(['Rebin', 'Integration']))
        self.declareProperty(WorkspaceProperty('OutputWorkspace', '', direction=Direction.Output))

    def PyExec(self):
        workspace = self.getProperty('InputWorkspace').value
        cut_dict = self.getProperty('CutAxis').value
        cut_axis = Axis(cut_dict['units'].value, cut_dict['start'].value, cut_dict['end'].value, cut_dict['step'].value,
                        cut_dict['e_unit'].value)
        int_dict = self.getProperty('IntegrationAxis').value
        int_axis = Axis(int_dict['units'].value, int_dict['start'].value, int_dict['end'].value, int_dict['step'].value,
                        int_dict['e_unit'].value)
        e_mode = self.getProperty('EMode').value
        PSD = self.getProperty('PSD').value
        norm_to_one = self.getProperty('NormToOne').value
        ignore_partial_overlaps = self.getProperty('IgnorePartialOverlaps').value
        algo = self.getProperty('Algorithm').value
        cut = compute_cut(workspace, cut_axis, int_axis, e_mode, PSD, norm_to_one, ignore_partial_overlaps, algo)
        if 'DeltaE' in cut_axis.units and cut_axis.scale != 1.:
            cut = TransformMD(InputWorkspace=cut, Scaling=[EnergyUnits(cut_axis.e_unit).factor_from_meV()])
        attribute_to_log({'axes': [cut_axis, int_axis], 'norm_to_one': norm_to_one, 'algorithm': algo}, cut)
        self.setProperty('OutputWorkspace', cut)

    def category(self):
        return 'MSlice'


def compute_cut(selected_workspace, cut_axis, integration_axis, e_mode, PSD, is_norm, ignore_partial_overlaps, algo):
    if PSD:
        cut = _compute_cut_PSD(selected_workspace, cut_axis, integration_axis, algo)
    else:
        cut = _compute_cut_nonPSD(selected_workspace, cut_axis, integration_axis, e_mode, ignore_partial_overlaps, algo)
    if is_norm:
        normalize_workspace(cut)
    return cut


def _compute_cut_PSD(selected_workspace, cut_axis, integration_axis, algo):
    # Do we need to pass ignore_partial_overlaps into here (PSD data)?
    cut_axis.units = cut_axis.units.replace('2Theta', 'Degrees')
    integration_axis.units = integration_axis.units.replace('2Theta', 'Degrees')
    fill_in_missing_input(cut_axis, selected_workspace)
    n_steps = get_number_of_steps(cut_axis)
    cut_binning = " ,".join(map(str, (cut_axis.units, cut_axis.start_meV, cut_axis.end_meV, n_steps)))
    integration_binning = integration_axis.units + "," + str(integration_axis.start_meV) + "," + \
        str(integration_axis.end_meV) + (",1" if 'Rebin' in algo else ",100")

    ws = BinMD(InputWorkspace=selected_workspace, AxisAligned="1", AlignedDim0=integration_binning,
               AlignedDim1=cut_binning, StoreInADS=False)
    if 'Integration' in algo:
        x0, x1 = (integration_axis.start_meV, integration_axis.end_meV)
        # 100 step is hard coded into the `integration_binning` string above
        norm_fac = (np.sum(ws.getNumEventsArray() != 0., axis=0) / 100) * (x1 - x0)
        ws = IntegrateMDHistoWorkspace(ws, P1Bin=[x0, x1], P2Bin=[], StoreInADS=False)
        ws.setSignalArray(ws.getSignalArray() * norm_fac)
        ws.setErrorSquaredArray(ws.getErrorSquaredArray() * np.square(norm_fac))
    return ws


def _compute_cut_nonPSD(selected_workspace, cut_axis, integration_axis, emode, ignore_partial_overlaps, algo):
    cut_binning = " ,".join(map(str, (cut_axis.start_meV, cut_axis.step_meV, cut_axis.end_meV)))
    int_binning = " ,".join(map(str, (integration_axis.start_meV, integration_axis.end_meV - integration_axis.start_meV,
                                      integration_axis.end_meV)))
    idx = 0 if 'Rebin' in algo else 1
    unit = 'DeltaE'
    name = '__MSL_EnergyTransfer'
    if is_momentum(cut_axis.units):
        ws_out = _cut_nonPSD_momentum(cut_binning, int_binning, emode, selected_workspace, ignore_partial_overlaps, algo)
        idx = 1
        unit = 'MomentumTransfer'
        name = '__MSL_|Q|'
    elif is_twotheta(cut_axis.units):
        ws_out = _cut_nonPSD_theta(int_binning, cut_binning, selected_workspace, ignore_partial_overlaps, algo)
        idx = 1 if 'Rebin' in algo else 0
        unit = 'Degrees'
        name = '__MSL_Theta'
    elif integration_axis.units == '|Q|':
        ws_out = _cut_nonPSD_momentum(int_binning, cut_binning, emode, selected_workspace, ignore_partial_overlaps, algo)
    else:
        ws_out = _cut_nonPSD_theta(cut_binning, int_binning, selected_workspace, algo)
    xdim = ws_out.getDimension(idx)
    extents = " ,".join(map(str, (xdim.getMinimum(), xdim.getMaximum())))

    # Hack to (deep) copy log data (ExperimentInfo)
    _tmpws = CreateSimulationWorkspace(Instrument='MAR', BinParams=[-1, 1, 1], UnitX='DeltaE', OutputWorkspace=name,
                                       EnableLogging=False)
    CopyLogs(ws_out, _tmpws, EnableLogging=False)
    AddSampleLog(_tmpws, LogName='Ei', LogText='3.', LogType='Number', EnableLogging=False)
    _tmpws = ConvertToMD(_tmpws, EnableLogging=False, StoreInADS=False, PreprocDetectorsWS='-',
                         QDimensions='|Q|', dEAnalysisMode='Direct')
    # TODO: Refactor the above code after Mantid framework has been changed to avoid the hack.
    # The above lines create an empty MD workspace with the same logs as the calculated Workspace2D output of the
    # cut algorithms. These logs are then copied (using copyExperimentInfos below) to the generated MDHistoWorkspace
    # Ideally we should be able to use the CopyLogs algorithm between a Workspace2D and MD workspace or
    # copyExperimentInfos should understand a Workspace2D input, either of which will need changes to Mantid.
    ws_out = CreateMDHistoWorkspace(SignalInput=ws_out.extractY(), ErrorInput=ws_out.extractE(), Dimensionality=1,
                                    Extents=extents, NumberOfBins=xdim.getNBins(), Names=name, Units=unit,
                                    StoreInADS=False, EnableLogging=False)
    ws_out.copyExperimentInfos(_tmpws)
    DeleteWorkspace(_tmpws, EnableLogging=False)

    return ws_out


def _cut_nonPSD_theta(ax1_binning, ax2_binning, selected_workspace, ignore_partial_overlaps, algo):
    # Pass ignore_partial_overlaps to the Rebin2D algo, and possibly Rebin too

    converted_nonpsd = ConvertSpectrumAxis(InputWorkspace=selected_workspace, Target='theta', StoreInADS=False)

    if 'Integration' in algo:
        ax1 = [float(x1) for x1 in ax1_binning.split(',')]
        ax2 = [float(x2) for x2 in ax2_binning.split(',')]
        if np.abs((ax1[2]-ax1[0]) - ax1[1]) < 0.0001:
            ws_out = Integration(InputWorkspace=converted_nonpsd, RangeLower=ax1[0], RangeUpper=ax1[2], EnableLogging=False)
            ws_out = Transpose(InputWorkspace=ws_out, EnableLogging=False)
            ws_out = Rebin(InputWorkspace=ws_out, Params=ax2_binning, EnableLogging=False)
        else:
            ws_out = Rebin(InputWorkspace=converted_nonpsd, Params=ax1_binning, EnableLogging=False)
            ws_out = Transpose(InputWorkspace=ws_out, EnableLogging=False)
            ws_out = Integration(InputWorkspace=ws_out, RangeLower=ax2[0], RangeUpper=ax2[2], EnableLogging=False)
    else:
        ws_out = Rebin2D(InputWorkspace=converted_nonpsd, Axis1Binning=ax1_binning, Axis2Binning=ax2_binning,
                         StoreInADS=False, UseFractionalArea=True, EnableLogging=False)
    return ws_out


def _cut_indirect_or_direct(q_binning, e_binning, emode, selected_workspace, ignore_partial_overlaps):
    # Pass ignore_partial_overlaps to the SofQW3 algorithm
    if 'Indirect' in emode and selected_workspace.run().hasProperty('Efix'):
        ws_out = SofQW3(InputWorkspace=selected_workspace, QAxisBinning=q_binning, EAxisBinning=e_binning, EMode=emode,
                        StoreInADS=False, EFixed=selected_workspace.run().getProperty('Efix').value, EnableLogging=False)
    else:
        ws_out = SofQW3(InputWorkspace=selected_workspace, OutputWorkspace='out', EMode=emode, QAxisBinning=q_binning,
                        EAxisBinning=e_binning, StoreInADS=False, EnableLogging=False)
    return ws_out


def _cut_nonPSD_momentum(q_binning, e_binning, emode, selected_workspace, ignore_partial_overlaps, algo):
    if 'Integration' in algo:
        qbins = [float(q) for q in q_binning.split(',')]
        ebins = [float(e) for e in e_binning.split(',')]
        if np.abs((qbins[2]-qbins[0]) - qbins[1]) < 0.0001:
            qbinstr = ','.join(map(str, [qbins[0], (qbins[2]-qbins[0])/100., qbins[2]]))
            ws_out = _cut_indirect_or_direct(qbinstr, e_binning, emode, selected_workspace, ignore_partial_overlaps)
            ws_out = Transpose(InputWorkspace=ws_out, EnableLogging=False)
            ws_out = Integration(InputWorkspace=ws_out, RangeLower=qbins[0], RangeUpper=qbins[2], EnableLogging=False)
        else:
            ebinstr = ','.join(map(str, [ebins[0], (ebins[2]-ebins[0])/100., ebins[2]]))
            ws_out = _cut_indirect_or_direct(q_binning, ebinstr, emode, selected_workspace, ignore_partial_overlaps)
            ws_out = Integration(InputWorkspace=ws_out, RangeLower=ebins[0], RangeUpper=ebins[2], EnableLogging=False)
    else:
        ws_out = _cut_indirect_or_direct(q_binning, e_binning, emode, selected_workspace, ignore_partial_overlaps)
    return ws_out
