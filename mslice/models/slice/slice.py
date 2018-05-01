from mantid.api import PythonAlgorithm, WorkspaceProperty, IMDEventWorkspace
from mantid.kernel import Direction, StringMandatoryValidator, PropertyManagerProperty
from mantid.simpleapi import BinMD, Rebin2D, ConvertSpectrumAxis, SofQW3


class Slice(PythonAlgorithm):

    def PyInit(self):
        self.declareProperty(WorkspaceProperty('InputWorkspace', "", direction=Direction.Input))
        self.declareProperty('XAxis', '')
        self.declareProperty('YAxis', '')
        self.declareProperty('EMode', 'Direct', StringMandatoryValidator())
        self.declareProperty('PSD', False)
        self.declareProperty('NormToOne', False)
        self.declareProperty(WorkspaceProperty('OutputWorkspace', '', direction=Direction.Output))

    def PyExec(self):
        workspace = self.getProperty('InputWorkspace')
        x_axis = self.getProperty('XAxis')
        y_axis = self.getProperty('YAxis')
        norm_to_one = self.getProperty('NormToOne')
        if self.getProperty('PSD').value:
            slice = self._compute_slice_PSD(workspace, x_axis, y_axis, norm_to_one)
        else:
            e_mode = self.getProperty('EMode')
            slice = self._compute_slice_nonPSD(workspace, x_axis, y_axis, e_mode, norm_to_one)
        self.setProperty('OutputWorkspace', slice)


    def category(self):
        return 'MSlice'

    def _compute_slice_PSD(self, workspace, x_axis, y_axis, norm_to_one):
        assert isinstance(workspace, IMDEventWorkspace)
        raw_ws = workspace.raw_ws
        self._fill_in_missing_input(x_axis, raw_ws)
        self._fill_in_missing_input(y_axis, raw_ws)
        n_x_bins = self._get_number_of_steps(x_axis)
        n_y_bins = self._get_number_of_steps(y_axis)
        x_dim_id = raw_ws.getDimensionIndexByName(x_axis.units)
        y_dim_id = raw_ws.getDimensionIndexByName(y_axis.units)
        x_dim = raw_ws.getDimension(x_dim_id)
        y_dim = raw_ws.getDimension(y_dim_id)
        xbinning = x_dim.getName() + "," + str(x_axis.start) + "," + str(x_axis.end) + "," + str(n_x_bins)
        ybinning = y_dim.getName() + "," + str(y_axis.start) + "," + str(y_axis.end) + "," + str(n_y_bins)
        return BinMD(InputWorkspace=raw_ws, AxisAligned="1", AlignedDim0=xbinning, AlignedDim1=ybinning,
                     StoreInADS=False)

    def _compute_slice_nonPSD(self, workspace, x_axis, y_axis, e_mode, norm_to_one):
        axes = [x_axis, y_axis]
        if x_axis.units == 'DeltaE':
            e_axis = 0
        elif y_axis.units == 'DeltaE':
            e_axis = 1
        else:
            raise RuntimeError('Cannot calculate slices without an energy axis')
        q_axis = (e_axis + 1) % 2
        ebin = '%f, %f, %f' % (axes[e_axis].start, axes[e_axis].step, axes[e_axis].end)
        qbin = '%f, %f, %f' % (axes[q_axis].start, axes[q_axis].step, axes[q_axis].end)
        if axes[q_axis].units == '|Q|':
            thisslice = SofQW3(InputWorkspace=workspace, QAxisBinning=qbin, EAxisBinning=ebin, EMode=e_mode,
                               StoreInADS=False)
        else:
            thisslice = ConvertSpectrumAxis(InputWorkspace=workspace, Target='Theta', StoreInADS=False)
            thisslice = Rebin2D(InputWorkspace=thisslice, Axis1Binning=ebin, Axis2Binning=qbin, StoreInADS=False)
        return thisslice
