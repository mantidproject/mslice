from __future__ import (absolute_import, division, print_function)

from mslice.models.workspacemanager.workspace_algorithms import propagate_properties
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.models.projection.powder.projection_calculator import ProjectionCalculator
from mslice.util.mantid import mantid_algorithms
from mslice.models.labels import DELTA_E_LABEL, MEV_LABEL, MOD_Q_LABEL, WAVENUMBER_LABEL, THETA_LABEL


class CLIProjectionCalculator(ProjectionCalculator):

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

    def calculate_projection(self, input_workspace, axis1, axis2, units):
        """Calculate the projection workspace AND return a python handle to it"""
        workspace = get_workspace_handle(input_workspace)
        if not workspace.is_PSD:
            raise RuntimeError('Cannot calculate projections for non-PSD workspaces')

        # can have Q-E or 2theta-E or their transpose.
        if axis1 != DELTA_E_LABEL and axis2 != DELTA_E_LABEL:
            raise NotImplementedError("Must have a '%s' axis" % DELTA_E_LABEL)
        if (axis1 == MOD_Q_LABEL or axis2 == MOD_Q_LABEL):
            projection_type='QE'
            output_workspace_name = workspace.name + ('_QE' if axis1 == MOD_Q_LABEL else '_EQ')
        elif (axis1 == THETA_LABEL or axis2 == THETA_LABEL):
            projection_type='Theta'
            output_workspace_name = workspace.name + ('_ThE' if axis1 == THETA_LABEL else '_ETh')
        else:
            raise NotImplementedError(" Axis '%s' not recognised. Must be '|Q|' or '2Theta'." % (axis1 if
                                      axis1 != DELTA_E_LABEL else axis2))
        if units == WAVENUMBER_LABEL:
            output_workspace_name += '_cm'

        new_ws = mantid_algorithms.MakeProjection(OutputWorkspace=output_workspace_name, InputWorkspace=workspace,
                                                  Axis1=axis1, Axis2=axis2, Units=units, EMode=workspace.e_mode,
                                                  Limits=workspace.limits['MomentumTransfer'],
                                                  ProjectionType=projection_type)
        propagate_properties(workspace, new_ws)
        return new_ws
