from __future__ import (absolute_import, division, print_function)

from mslice.models.workspacemanager.mantid_workspace_provider import get_workspace_handle, propagate_properties, run_algorithm
from mslice.models.projection.powder.projection_calculator import ProjectionCalculator
import mantid.simpleapi as s_api
from ...labels import DELTA_E_LABEL, MEV_LABEL, MOD_Q_LABEL, WAVENUMBER_LABEL, THETA_LABEL


class MantidProjectionCalculator(ProjectionCalculator):

    def available_axes(self):
        return [MOD_Q_LABEL, THETA_LABEL, DELTA_E_LABEL]

    def available_units(self):
        return [MEV_LABEL, WAVENUMBER_LABEL]

    def validate_workspace(self, ws):
        workspace = get_workspace_handle(ws)
        try:
            axes = [workspace.raw_ws.getAxis(0), workspace.raw_ws.getAxis(1)]
            if not all([ax.isSpectra() or ax.getUnit().unitID() == 'DeltaE' for ax in axes]):
                raise AttributeError
        except (AttributeError, IndexError):
            raise TypeError('Input workspace for projection calculation must be a reduced '
                            'data workspace with a spectra and energy transfer axis.')

    def calculate_projection(self, input_workspace_name, axis1, axis2, units):
        """Calculate the projection workspace AND return a python handle to it"""
        workspace = get_workspace_handle(input_workspace_name)
        if not workspace.is_PSD:
            raise RuntimeError('Cannot calculate projections for non-PSD workspaces')

        # can have Q-E or 2theta-E or their transpose.
        if axis1 != DELTA_E_LABEL and axis2 != DELTA_E_LABEL:
            raise NotImplementedError('Must have an energy axis')
        if (axis1 == MOD_Q_LABEL or axis2 == MOD_Q_LABEL):
            projection_type='QE'
            output_workspace_name = input_workspace_name + ('_QE' if axis1 == MOD_Q_LABEL else '_EQ')
        elif (axis1 == THETA_LABEL or axis2 == THETA_LABEL):
            projection_type='Theta'
            output_workspace_name = input_workspace_name + ('_ThE' if axis1 == THETA_LABEL else '_ETh')
        else:
            raise NotImplementedError('Only Q or Theta axes supported')
        if units == WAVENUMBER_LABEL:
            output_workspace_name += '_cm'

        new_ws = run_algorithm(s_api.MakeProjection, output_name=output_workspace_name, InputWorkspace=workspace, Axis1=axis1,
                         Axis2=axis2, Units=units, EMode=workspace.e_mode, Limits=workspace.limits['MomentumTransfer'],
                         ProjectionType=projection_type)
        propagate_properties(workspace, new_ws)
        return new_ws
