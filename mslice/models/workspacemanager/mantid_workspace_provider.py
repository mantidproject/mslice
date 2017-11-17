"""A concrete implementation of a WorkspaceProvider

It uses mantid to perform the workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)
from mantid.simpleapi import (AnalysisDataService, DeleteWorkspace, Load,
                              RenameWorkspace, SaveNexus, SaveMD, MergeMD)
from mantid.api import IMDEventWorkspace, IMDHistoWorkspace, Workspace
import numpy as np
from scipy import constants

from .workspace_provider import WorkspaceProvider

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------


class MantidWorkspaceProvider(WorkspaceProvider):
    def __init__(self):
        # Stores various parameters of workspaces not stored by Mantid
        self._EfDefined = {}
        self._limits = {}
        self._cutParameters = {}

    def get_workspace_names(self):
        return AnalysisDataService.getObjectNames()

    def delete_workspace(self, workspace):
        ws = DeleteWorkspace(Workspace=workspace)
        if workspace in self._EfDefined:
            del self._EfDefined[workspace]
        if workspace in self._limits:
            del self._limits[workspace]
        return ws

    def get_limits(self, workspace, axis):
        """Determines the limits of the data and minimum step size"""
        if workspace not in self._limits:
            self._processLoadedWSLimits(workspace)
        # If we cannot get the step size from the data, use the old 1/100 steps.
        if axis in self._limits[workspace]:
            return self._limits[workspace][axis]
        else:
            ws_h = self.get_workspace_handle(workspace)
            dim = ws_h.getDimension(ws_h.getDimensionIndexByName(axis))
            minimum = dim.getMinimum()
            maximum = dim.getMaximum()
            return minimum, maximum, (maximum - minimum) / 100.

    def _processEfixed(self, workspace):
        """Checks whether the fixed energy is defined for this workspace"""
        ws_name = workspace if isinstance(workspace, str) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(ws_name)
        try:
            [self._get_ws_EFixed(ws_h, ws_h.getDetector(i).getID()) for i in range(ws_h.getNumberHistograms())]
            self._EfDefined[ws_name] = True
        except RuntimeError:
            self._EfDefined[ws_name] = False

    def _processLoadedWSLimits(self, workspace):
        """ Processes an (angle-deltaE) workspace to get the limits and step size in angle, energy and |Q| """
        ws_name = workspace if isinstance(workspace, str) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(workspace)
        # For cases, e.g. indirect, where EFixed has not been set yet, return calculate later.
        efix = self._get_EFixed_for_limits(ws_h)
        if efix is None:
            self._limits[ws_name] = {}
            return
        en = ws_h.getAxis(0).extractValues()
        # Defines some conversion factors
        E2q = 2. * constants.m_n / (constants.hbar ** 2)  # Energy to (neutron momentum)^2 (==2m_n/hbar^2)
        meV2J = constants.e / 1000.                       # meV to Joules
        m2A = 1.e10                                       # metres to Angstrom
        if ws_name not in self._limits:
            self._limits[ws_name] = {}
        theta = self._get_theta_for_limits(ws_h)
        # Use |Q| at elastic line to get minimum and step size
        qmin, qmax, qstep = tuple(np.sqrt(E2q * 2 * efix * (1 - np.cos(theta)) * meV2J) / m2A)
        # Use minimum energy (Direct geometry) or maximum energy (Indirect) to get qmax
        emax = -np.min(en) if (str(ws_h.getEMode()) == 'Direct') else np.max(en)
        qmax = np.sqrt(E2q * (2 * efix + emax - 2 * np.sqrt(efix * (efix + emax)) * np.cos(theta[1])) * meV2J) / m2A
        self._limits[ws_name]['MomentumTransfer'] = [qmin - qstep, qmax + qstep, qstep]
        self._limits[ws_name]['|Q|'] = self._limits[ws_name]['MomentumTransfer']  # ConvertToMD renames it(!)
        self._limits[ws_name]['Degrees'] = theta * 180 / np.pi
        self._limits[ws_name]['DeltaE'] = [np.min(en), np.max(en), np.mean(np.diff(en))]

    def _get_EFixed_for_limits(self, ws_handle):
        try:
            efix = self._get_ws_EFixed(ws_handle, ws_handle.getDetector(0).getID())
        except RuntimeError:  # Efixed not defined
            # This could occur for malformed NXSPE without the instrument name set.
            # LoadNXSPE then sets EMode to 'Elastic' and getEFixed fails.
            if ws_handle.run().hasProperty('Ei'):
                efix = ws_handle.run().getProperty('Ei').value
            else:
                return None
        except AttributeError:  # Wrong workspace type (e.g. cut)
            return None
        # Checks that loaded data is in energy transfer.
        enAxis = ws_handle.getAxis(0)
        if 'DeltaE' not in enAxis.getUnit().unitID():
            return None
        return efix

    def _get_theta_for_limits(self, ws_handle):
        # Don't parse all spectra in cases where there are alot to save time.
        num_hist = ws_handle.getNumberHistograms()
        if num_hist > 500:
            theta = [ws_handle.detectorTwoTheta(ws_handle.getDetector(i))
                     for i in range(0, num_hist, int(num_hist / 500))]
        else:
            theta = [ws_handle.detectorTwoTheta(ws_handle.getDetector(i)) for i in range(num_hist)]
        # Rounds the differences to avoid pixels with same 2theta. Implies min limit of ~0.1 degrees
        thdiff = np.diff(np.round(np.sort(theta)*573)/573)
        return np.array([np.min(theta), np.max(theta), np.min(thdiff[np.where(thdiff>0)])])

    def load(self, filename, output_workspace):
        ws = Load(Filename=filename, OutputWorkspace=output_workspace)
        if self.get_EMode(output_workspace) == 'Indirect':
            self._processEfixed(output_workspace)
        self._processLoadedWSLimits(output_workspace)
        return ws

    def rename_workspace(self, selected_workspace, new_name):
        ws = RenameWorkspace(InputWorkspace=selected_workspace, OutputWorkspace=new_name)
        if selected_workspace in self._limits:
            self._limits[new_name] = self._limits.pop(selected_workspace)
        if selected_workspace in self._EfDefined:
            self._EfDefined[new_name] = self._EfDefined.pop(selected_workspace)
        if selected_workspace in self._cutParameters:
            self._cutParameters[new_name] = self._cutParameters.pop(selected_workspace)
        return ws

    def combine_workspace(self, selected_workspaces, new_name):
        ws = MergeMD(InputWorkspaces=selected_workspaces, OutputWorkspace=new_name)
        # Use precalculated step size, otherwise get limits directly from workspace
        ax1 = ws.getDimension(0)
        ax2 = ws.getDimension(1)
        step1 = []
        step2 = []
        for input_workspace in selected_workspaces:
            step1.append(self.get_limits(input_workspace, ax1.name)[2])
            step2.append(self.get_limits(input_workspace, ax2.name)[2])
        if new_name not in self._limits.keys():
            self._limits[new_name] = {}
        self._limits[new_name][ax1.name] = [ax1.getMinimum(), ax1.getMaximum(), np.max(step1)]
        self._limits[new_name][ax2.name] = [ax2.getMinimum(), ax2.getMaximum(), np.max(step2)]
        return ws

    def save_nexus(self, workspace, path):
        workspace_handle = self.get_workspace_handle(workspace)
        if isinstance(workspace_handle, IMDEventWorkspace) or isinstance(workspace_handle, IMDHistoWorkspace):
            SaveMD(InputWorkspace=workspace, Filename=path)
        else:
            SaveNexus(InputWorkspace=workspace, Filename=path)

    def is_pixel_workspace(self, workspace_name):
        from mantid.api import IMDEventWorkspace
        workspace = self.get_workspace_handle(workspace_name)
        return isinstance(workspace, IMDEventWorkspace)

    def get_workspace_handle(self, workspace_name):
        """"Return handle to workspace given workspace_name_as_string"""
        # if passed a workspace handle return the handle
        if isinstance(workspace_name, Workspace):
            return workspace_name
        return AnalysisDataService[workspace_name]

    def get_workspace_name(self, workspace):
        """Returns the name of a workspace given the workspace handle"""
        if isinstance(workspace, str):
            return workspace
        return workspace.name()

    def get_EMode(self, workspace):
        """Returns the energy analysis mode (direct or indirect of a workspace)"""
        if isinstance(workspace, str):
            workspace_handle = self.get_workspace_handle(workspace)
        else:
            workspace_handle = workspace
        emode = str(self._get_ws_EMode(workspace_handle))
        if emode == 'Elastic':
            # Work-around for older versions of Mantid which does not set instrument name
            # in NXSPE files, so LoadNXSPE does not know if it is direct or indirect data
            ei_log = workspace_handle.run().getProperty('Ei').value
            emode = 'Indirect' if np.isnan(ei_log) else 'Direct'
        return emode

    def _get_ws_EMode(self, ws_handle):
        if isinstance(ws_handle, IMDHistoWorkspace) or isinstance(ws_handle, IMDEventWorkspace):
            def get_emode(e):
                ws_handle.getExperimentInfo(e).getEMode()
            return self._get_exp_info_using(ws_handle, get_emode, "Workspace contains different EModes")
        else:
            return ws_handle.getEMode()

    def _get_ws_EFixed(self, ws_handle, detector):
        if isinstance(ws_handle, IMDHistoWorkspace) or isinstance(ws_handle, IMDEventWorkspace):
            def get_efixed(e):
                ws_handle.getExperimentInfo(e).getEFixed(detector)
            return self._get_exp_info_using(ws_handle, get_efixed, "Workspace contains different EFixed values")
        else:
            return ws_handle.getEFixed(detector)

    def _get_exp_info_using(self, ws_handle, get_exp_info, error_string):
        """get data from MultipleExperimentInfo. Returns None if ExperimentInfo is not found"""
        prev = None
        for exp in range(ws_handle.getNumExperimentInfo()):
            exp_value = get_exp_info(exp)
            if prev is not None:
                if exp_value != prev:
                    raise ValueError(error_string)
            prev = exp_value
        return prev

    def has_efixed(self, workspace):
        return self._EfDefined[workspace if isinstance(workspace, str) else self.get_workspace_name(workspace)]

    def set_efixed(self, workspace, Ef):
        """Sets (overides) the fixed energy for all detectors (spectra) of this workspace"""
        ws_name = workspace if isinstance(workspace, str) else self.get_workspace_name(workspace)
        ws_handle = self.get_workspace_handle(ws_name)
        for idx in range(ws_handle.getNumberHistograms()):
            ws_handle.setEFixed(ws_handle.getDetector(idx).getID(), Ef)

    def propagate_properties(self, old_workspace, new_workspace):
        """Propagates MSlice only properties of workspaces, e.g. limits"""
        if old_workspace in self._EfDefined:
            self._EfDefined[new_workspace] = self._EfDefined[old_workspace]
        if old_workspace in self._limits:
            self._limits[new_workspace] = self._limits[old_workspace]

    def getComment(self, workspace):
        if hasattr(workspace, 'getComment'):
            return workspace.getComment()
        ws_handle = self.get_workspace_handle(workspace)
        return ws_handle.getComment()

    def setCutParameters(self, workspace, axis, parameters):
        if workspace not in self._cutParameters:
            self._cutParameters[workspace] = dict()
        self._cutParameters[workspace][axis] = parameters
        self._cutParameters[workspace]['previous_axis'] = axis

    def getCutParameters(self, workspace, axis=None):
        if workspace in self._cutParameters:
            if axis is not None:
                if axis in self._cutParameters[workspace]:
                    return self._cutParameters[workspace][axis], axis
                else:
                    return None, None
            else:
                prev_axis = self._cutParameters[workspace]['previous_axis']
                return self._cutParameters[workspace][prev_axis], prev_axis
        return None, None

    def isAxisSaved(self, workspace, axis):
        if workspace in self._cutParameters:
            return True if axis in self._cutParameters[workspace] else False
        return False
