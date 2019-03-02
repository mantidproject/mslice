from mantid.api import PythonAlgorithm, WorkspaceProperty, IMDEventWorkspace
from mantid.kernel import Direction, StringMandatoryValidator, PropertyManagerProperty
from mantid.simpleapi import BinMD, Rebin2D, ConvertSpectrumAxis, SofQW3, TransformMD, ScaleX
from mslice.models.alg_workspace_ops import get_number_of_steps
from mslice.models.axis import Axis
from mslice.models.units import EnergyUnits
from mslice.workspace.helperfunctions import attribute_to_log


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
        self.declareProperty(WorkspaceProperty('OutputWorkspace', '', direction=Direction.Output))

    def PyExec(self):
        workspace = self.getProperty('InputWorkspace').value
        x_dict = self.getProperty('XAxis').value
        x_axis = Axis(x_dict['units'].value, x_dict['start'].value, x_dict['end'].value, x_dict['step'].value, x_dict['e_unit'].value)
        y_dict = self.getProperty('YAxis').value
        y_axis = Axis(y_dict['units'].value, y_dict['start'].value, y_dict['end'].value, y_dict['step'].value, y_dict['e_unit'].value)
        e_axis = 0 if 'DeltaE' in x_axis.units else (1 if 'DeltaE' in y_axis.units else None)
        if e_axis is not None:
            e_scale = EnergyUnits(x_axis.e_unit if e_axis == 0 else y_axis.e_unit).factor_from_meV()
        norm_to_one = self.getProperty('NormToOne')
        if self.getProperty('PSD').value:
            slice = self._compute_slice_PSD(workspace, x_axis, y_axis, norm_to_one)
            if e_axis is not None and e_scale != 1.:
                scale = [1., e_scale] if e_axis == 1 else [e_scale, 1.]
                slice = TransformMD(InputWorkspace=slice, Scaling=scale)
        else:
            e_mode = self.getProperty('EMode').value
            slice = self._compute_slice_nonPSD(workspace, x_axis, y_axis, e_mode, norm_to_one)
            # For non-PSD data one of the axes *must* be DeltaE
            if e_scale != 1.:
                slice = ScaleX(InputWorkspace=slice, Factor=e_scale, Operation='Multiply', StoreInADS=False)
        attribute_to_log({'axes': [x_axis, y_axis]}, slice)
        self.setProperty('OutputWorkspace', slice)

    def category(self):
        return 'MSlice'

    def _compute_slice_PSD(self, workspace, x_axis, y_axis, norm_to_one):
        assert isinstance(workspace, IMDEventWorkspace)
        n_x_bins = get_number_of_steps(x_axis)
        n_y_bins = get_number_of_steps(y_axis)
        x_dim_id = self.dimension_index(workspace, x_axis)
        y_dim_id = self.dimension_index(workspace, y_axis)
        x_dim = workspace.getDimension(x_dim_id)
        y_dim = workspace.getDimension(y_dim_id)
        xbinning = x_dim.getName() + "," + str(x_axis.start_meV) + "," + str(x_axis.end_meV) + "," + str(n_x_bins)
        ybinning = y_dim.getName() + "," + str(y_axis.start_meV) + "," + str(y_axis.end_meV) + "," + str(n_y_bins)
        return BinMD(InputWorkspace=workspace, AxisAligned="1", AlignedDim0=xbinning, AlignedDim1=ybinning,
                     StoreInADS=False)

    def dimension_index(self, workspace, axis):
        try:
            return workspace.getDimensionIndexByName(axis.units)
        except RuntimeError as e:
            if axis.units == '2Theta':
                return workspace.getDimensionIndexByName('Degrees')
            else:
                raise e

    def _compute_slice_nonPSD(self, workspace, x_axis, y_axis, e_mode, norm_to_one):
        axes = [x_axis, y_axis]
        if x_axis.units == 'DeltaE':
            e_axis = 0
        elif y_axis.units == 'DeltaE':
            e_axis = 1
        else:
            raise RuntimeError('Cannot calculate slices without an energy axis')
        q_axis = (e_axis + 1) % 2
        ebin = '%f, %f, %f' % (axes[e_axis].start_meV, axes[e_axis].step_meV, axes[e_axis].end_meV)
        qbin = '%f, %f, %f' % (axes[q_axis].start_meV, axes[q_axis].step_meV, axes[q_axis].end_meV)
        if axes[q_axis].units == '|Q|':
            thisslice = SofQW3(InputWorkspace=workspace, QAxisBinning=qbin, EAxisBinning=ebin, EMode=e_mode,
                               StoreInADS=False)
        elif axes[q_axis].units == '2Theta':
            thisslice = ConvertSpectrumAxis(InputWorkspace=workspace, Target='Theta', StoreInADS=False)
            thisslice = Rebin2D(InputWorkspace=thisslice, Axis1Binning=ebin, Axis2Binning=qbin, StoreInADS=False)
        else:
            raise RuntimeError("axis %s not recognised, must be '|Q|' or '2Theta'" % axes[q_axis].units)
        return thisslice
