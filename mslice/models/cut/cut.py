
from mantid.api import PythonAlgorithm, WorkspaceProperty
from mantid.kernel import Direction, PropertyManagerProperty, StringMandatoryValidator
from mantid.simpleapi import BinMD, ConvertSpectrumAxis, CreateMDHistoWorkspace, Rebin2D, SofQW3

from mslice.models.alg_workspace_ops import fill_in_missing_input, get_number_of_steps
from mslice.models.axis import Axis
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
        self.declareProperty(WorkspaceProperty('OutputWorkspace', '', direction=Direction.Output))

    def PyExec(self):
        workspace = self.getProperty('InputWorkspace').value
        cut_dict = self.getProperty('CutAxis').value
        cut_axis = Axis(cut_dict['units'].value, cut_dict['start'].value, cut_dict['end'].value, cut_dict['step'].value)
        int_dict = self.getProperty('IntegrationAxis').value
        int_axis = Axis(int_dict['units'].value, int_dict['start'].value, int_dict['end'].value, int_dict['step'].value)
        e_mode = self.getProperty('EMode').value
        PSD = self.getProperty('PSD').value
        norm_to_one = self.getProperty('NormToOne').value
        cut = compute_cut(workspace, cut_axis, int_axis, e_mode, PSD, norm_to_one)
        self.setProperty('OutputWorkspace', cut)

    def category(self):
        return 'MSlice'

def compute_cut(selected_workspace, cut_axis, integration_axis, e_mode, PSD, is_norm):
    if PSD:
        cut = _compute_cut_PSD(selected_workspace, cut_axis, integration_axis)
    else:
        cut = _compute_cut_nonPSD(selected_workspace, cut_axis, integration_axis, e_mode)
    if is_norm:
        normalize_workspace(cut)
    return cut


def _compute_cut_PSD(selected_workspace, cut_axis, integration_axis):
    fill_in_missing_input(cut_axis, selected_workspace)
    n_steps = get_number_of_steps(cut_axis)
    cut_binning = " ,".join(map(str, (cut_axis.units, cut_axis.start, cut_axis.end, n_steps)))
    integration_binning = integration_axis.units + "," + str(integration_axis.start) + "," + \
        str(integration_axis.end) + ",1"

    return BinMD(InputWorkspace=selected_workspace, AxisAligned="1", AlignedDim1=integration_binning,
                 AlignedDim0=cut_binning, StoreInADS=False)


def _compute_cut_nonPSD(selected_workspace, cut_axis, integration_axis, emode):
    cut_binning = " ,".join(map(str, (cut_axis.start, cut_axis.step, cut_axis.end)))
    int_binning = " ,".join(map(str, (integration_axis.start, integration_axis.end - integration_axis.start,
                                      integration_axis.end)))
    idx = 0
    unit = 'DeltaE'
    name = 'EnergyTransfer'
    if cut_axis.units == '|Q|':
        ws_out = _cut_nonPSD_momentum(cut_binning, int_binning, emode, selected_workspace)
        idx = 1
        unit = 'MomentumTransfer'
        name = '|Q|'
    elif cut_axis.units == 'Degrees':
        ws_out = _cut_nonPSD_theta(cut_binning, int_binning, selected_workspace)
        idx = 1
        unit = 'Degrees'
        name = 'Theta'
    elif integration_axis.units == '|Q|':
        ws_out = _cut_nonPSD_momentum(int_binning, cut_binning, emode, selected_workspace)
    else:
        ws_out = _cut_nonPSD_theta(int_binning, cut_binning, selected_workspace)
    xdim = ws_out.getDimension(idx)
    extents = " ,".join(map(str, (xdim.getMinimum(), xdim.getMaximum())))
    return CreateMDHistoWorkspace(SignalInput=ws_out.extractY(), ErrorInput=ws_out.extractE(), Dimensionality=1,
                                  Extents=extents, NumberOfBins=xdim.getNBins(), Names=name, Units=unit,
                                  StoreInADS=False)


def _cut_nonPSD_theta(cut_binning, int_binning, selected_workspace):

    converted_nonpsd = ConvertSpectrumAxis( OutputWorkspace='__convToTheta', InputWorkspace=selected_workspace,
                                            Target='theta', StoreInADS=False)

    ws_out = Rebin2D(InputWorkspace=converted_nonpsd, Axis1Binning=int_binning, Axis2Binning=cut_binning,
                     StoreInADS=False)
    return ws_out


def _cut_nonPSD_momentum(q_binning, e_binning, emode, selected_workspace):
    ws_out = SofQW3(InputWorkspace=selected_workspace, OutputWorkspace='out', EMode=emode, QAxisBinning=q_binning,
                    EAxisBinning=e_binning, StoreInADS=False)
    return ws_out
