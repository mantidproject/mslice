from mantid.api import PythonAlgorithm, WorkspaceProperty, IMDEventWorkspace
from mantid.kernel import Direction, StringMandatoryValidator, PropertyManagerProperty
from mantid.simpleapi import BinMD, Rebin2D, ConvertSpectrumAxis, SofQW3, TransformMD, ScaleX
from mslice.models.alg_workspace_ops import get_number_of_steps
from mslice.models.axis import Axis


class Slice(PythonAlgorithm):

    def PyInit(self):
        self.declareProperty(WorkspaceProperty('InputWorkspace', "", direction=Direction.Input))
        self.declareProperty(PropertyManagerProperty('XAxis', {}, direction=Direction.Input),
                             doc='MSlice Axis object as a dictionary')
        self.declareProperty(PropertyManagerProperty('YAxis', {}, direction=Direction.Input),
                             doc='MSlice Axis object as a dictionary')
        self.declareProperty('EMode', 'Direct', StringMandatoryValidator())
        self.declareProperty('PSD', False)
        self.declareProperty('NormToOne', False)
        self.declareProperty('EnergyUnit', 'meV')
        self.declareProperty(WorkspaceProperty('OutputWorkspace', '', direction=Direction.Output))

    def PyExec(self):
        workspace = self.getProperty('InputWorkspace').value
        x_dict = self.getProperty('XAxis').value
        x_axis = Axis(x_dict['units'].value, x_dict['start'].value, x_dict['end'].value, x_dict['step'].value)
        y_dict = self.getProperty('YAxis').value
        y_axis = Axis(y_dict['units'].value, y_dict['start'].value, y_dict['end'].value, y_dict['step'].value)
        norm_to_one = self.getProperty('NormToOne')
        e_unit = self.getProperty('EnergyUnit').value
        if self.getProperty('PSD').value:
            slice = self._compute_slice_PSD(workspace, x_axis, y_axis, norm_to_one, e_unit)
        else:
            e_mode = self.getProperty('EMode').value
            slice = self._compute_slice_nonPSD(workspace, x_axis, y_axis, e_mode, norm_to_one, e_unit)
        self.setProperty('OutputWorkspace', slice)

    def category(self):
        return 'MSlice'

    def _compute_slice_PSD(self, workspace, x_axis, y_axis, norm_to_one, e_unit):
        assert isinstance(workspace, IMDEventWorkspace)
        n_x_bins = get_number_of_steps(x_axis)
        n_y_bins = get_number_of_steps(y_axis)
        x_dim_id = self.dimension_index(workspace, x_axis)
        y_dim_id = self.dimension_index(workspace, y_axis)
        x_dim = workspace.getDimension(x_dim_id)
        y_dim = workspace.getDimension(y_dim_id)
        x_name = x_dim.getName()
        y_name = y_dim.getName()
        e_axis = 0 if 'DeltaE' in x_name else (1 if 'DeltaE' in y_name else None)
        xbinning = x_name + "," + str(x_axis.start) + "," + str(x_axis.end) + "," + str(n_x_bins)
        ybinning = y_name + "," + str(y_axis.start) + "," + str(y_axis.end) + "," + str(n_y_bins)
        if 'cm' in e_unit and e_axis is not None:
            if e_axis == 0:
                xbinning = x_name + "," + str(x_axis.start / 8.06544) + "," + str(x_axis.end / 8.06544) + "," + str(n_x_bins)
            else:
                ybinning = y_name + "," + str(y_axis.start / 8.06544) + "," + str(y_axis.end / 8.06544) + "," + str(n_y_bins)
        thisslice = BinMD(InputWorkspace=workspace, AxisAligned="1", AlignedDim0=xbinning, AlignedDim1=ybinning,
                          StoreInADS=False)
        if 'cm' in e_unit and e_axis is not None:
            scale = [1, 8.06554] if 'DeltaE' in y_name else [8.06544, 1]
            thisslice = TransformMD(InputWorkspace=thisslice, Scaling=scale)
            thisslice.setComment('MSlice_in_wavenumber')
        return thisslice

    def dimension_index(self, workspace, axis):
        try:
            return workspace.getDimensionIndexByName(axis.units)
        except RuntimeError as e:
            if axis.units == '2Theta':
                return workspace.getDimensionIndexByName('Degrees')
            else:
                raise e

    def _compute_slice_nonPSD(self, workspace, x_axis, y_axis, e_mode, norm_to_one, e_unit):
        axes = [x_axis, y_axis]
        if x_axis.units == 'DeltaE':
            e_axis = 0
        elif y_axis.units == 'DeltaE':
            e_axis = 1
        else:
            raise RuntimeError('Cannot calculate slices without an energy axis')
        q_axis = (e_axis + 1) % 2
        if 'cm' in e_unit:
            ebin = '%f, %f, %f' % (axes[e_axis].start / 8.06554, axes[e_axis].step / 8.06554, axes[e_axis].end / 8.06554)
        else:
            ebin = '%f, %f, %f' % (axes[e_axis].start, axes[e_axis].step, axes[e_axis].end)
        qbin = '%f, %f, %f' % (axes[q_axis].start, axes[q_axis].step, axes[q_axis].end)
        if axes[q_axis].units == '|Q|':
            thisslice = SofQW3(InputWorkspace=workspace, QAxisBinning=qbin, EAxisBinning=ebin, EMode=e_mode,
                               StoreInADS=False)
        elif axes[q_axis].units == '2Theta':
            thisslice = ConvertSpectrumAxis(InputWorkspace=workspace, Target='Theta', StoreInADS=False)
            thisslice = Rebin2D(InputWorkspace=thisslice, Axis1Binning=ebin, Axis2Binning=qbin, StoreInADS=False)
        else:
            raise RuntimeError("axis %s not recognised, must be '|Q|' or '2Theta'" % axes[q_axis].units)
        if 'cm' in e_unit:
            thisslice = ScaleX(InputWorkspace=thisslice, Factor=8.06554, Operation='Multiply', StoreInADS=False)
            thisslice.setComment('MSlice_in_wavenumber')
        return thisslice
