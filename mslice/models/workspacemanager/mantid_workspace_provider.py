"""A concrete implementation of a WorkspaceProvider

It uses mantid to perform the workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from mantid.simpleapi import (AnalysisDataService, DeleteWorkspace, Load,
                              RenameWorkspace, SaveNexus, SaveMD)
from mantid.api import IMDWorkspace, Workspace
import numpy as np
from scipy import constants

from .workspace_provider import WorkspaceProvider
from mslice.workspace.workspace import Workspace as WorkspaceWrapper
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.histogram_workspace import HistogramWorkspace

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class MantidWorkspaceProvider(WorkspaceProvider):
    def __init__(self):
        # Stores various parameters of workspaces not stored by Mantid
        self._EfDefined = {}
        self._limits = {}

    def get_workspace_names(self):
        return AnalysisDataService.getObjectNames()

    def delete_workspace(self, workspace):
        ws = DeleteWorkspace(Workspace=workspace)
        if workspace in self._EfDefined.keys():
            del self._EfDefined[workspace]
        if workspace in self._limits.keys():
            del self._limits[workspace]
        return ws

    def _processEfixed(self, workspace):
        """Checks whether the fixed energy is defined for this workspace"""
        ws_name = workspace if isinstance(workspace, basestring) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(ws_name)
        try:
            [ws_h.getEFixed(ws_h.getDetector(i).getID()) for i in range(ws_h.getNumberHistograms())]
            self._EfDefined[ws_name] = True
        except RuntimeError:
            self._EfDefined[ws_name] = False

    def _processLoadedWSLimits(self, workspace):
        """ Processes an (angle-deltaE) workspace to get the limits and step size in angle, energy and |Q| """
        ws_name = workspace if isinstance(workspace, basestring) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(workspace)
        # For cases, e.g. indirect, where EFixed has not been set yet, return calculate later.
        try:
            efix = ws_h.getEFixed(ws_h.getDetector(0).getID())
        except RuntimeError:   # Efixed not defined
            # This could occur for malformed NXSPE without the instrument name set.
            # LoadNXSPE then sets EMode to 'Elastic' and getEFixed fails.
            if ws_h.run().hasProperty('Ei'):
                efix = ws_h.run().getProperty('Ei').value
            else:
                self._limits[ws_name] = None
                return
        except AttributeError: # Wrong workspace type (e.g. cut)
            self._limits[ws_name] = {}
            return
        # Checks that loaded data is in energy transfer.
        enAxis = ws_h.getAxis(0)
        if 'DeltaE' not in enAxis.getUnit().unitID():
            self._limits[ws_name] = None
            return
        en = ws_h.getAxis(0).extractValues()
        # Don't parse all spectra in cases where there are alot to save time.
        numHist = ws_h.getNumberHistograms()
        if numHist > 500:
            theta = [ws_h.detectorTwoTheta(ws_h.getDetector(i)) for i in range(0, numHist, int(numHist/500))]
        else:
            theta = [ws_h.detectorTwoTheta(ws_h.getDetector(i)) for i in range(numHist)]
        # Defines some conversion factors
        E2q = 2. * constants.m_n / (constants.hbar ** 2)  # Energy to (neutron momentum)^2 (==2m_n/hbar^2)
        meV2J = constants.e / 1000.                       # meV to Joules
        m2A = 1.e10                                       # metres to Angstrom
        if ws_name not in self._limits.keys():
            self._limits[ws_name] = {}
        th = np.array([np.min(theta), np.max(theta), np.mean(np.diff(theta))])
        # Use |Q| at elastic line to get minimum and step size
        qmin, qmax, qstep = tuple(np.sqrt(E2q * 2 * efix * (1 - np.cos(th)) * meV2J) / m2A)
        # Use minimum energy (Direct geometry) or maximum energy (Indirect) to get qmax
        emax = -np.min(en) if (str(ws_h.getEMode()) == 'Direct') else np.max(en)
        qmax = np.sqrt(E2q * (2 * efix + emax - 2 * np.sqrt(efix * (efix + emax)) * np.cos(th[1])) * meV2J) / m2A
        self._limits[ws_name]['MomentumTransfer'] = [qmin - qstep, qmax + qstep, qstep]
        self._limits[ws_name]['|Q|'] = self._limits[ws_name]['MomentumTransfer'] # ConverToMD renames it(!)
        self._limits[ws_name]['Degrees'] = th * 180 / np.pi
        self._limits[ws_name]['DeltaE'] = [np.min(en), np.max(en), np.mean(np.diff(en))]

    def load(self, filename, output_workspace):
        ws = Load(Filename=filename, OutputWorkspace=output_workspace)
        if self.get_emode(output_workspace) == 'Indirect':
            self._processEfixed(output_workspace)
        self._processLoadedWSLimits(output_workspace)
        return ws

    def rename_workspace(self, selected_workspace, new_name):
        ws = RenameWorkspace(InputWorkspace=selected_workspace, OutputWorkspace=new_name)
        if selected_workspace in self._limits.keys():
            self._limits[new_name] = self._limits.pop(selected_workspace)
        if selected_workspace in self._EfDefined.keys():
            self._EfDefined[new_name] = self._EfDefined.pop(selected_workspace)
        return ws

    def save_nexus(self, workspace, path):
        workspace_handle = self.get_workspace_handle(workspace)
        if isinstance(workspace_handle, IMDWorkspace):
            SaveMD(InputWorkspace=workspace, Filename=path)
        else:
            SaveNexus(InputWorkspace=workspace, Filename=path)

    def get_workspace_handle(self, workspace_name):
        """"Return handle to workspace given workspace_name_as_string"""
        # if passed a workspace handle return the handle
        if isinstance(workspace_name, Workspace):
            return workspace_name
        return AnalysisDataService[workspace_name]

    def get_workspace_name(self, workspace):
        """Returns the name of a workspace given the workspace handle"""
        if isinstance(workspace, basestring):
            return workspace
        return workspace.name()

    def get_emode(self, workspace):
        """Returns the energy analysis mode (direct or indirect of a workspace)"""
        if isinstance(workspace, basestring):
            workspace_handle = self.get_workspace_handle(workspace)
        else:
            workspace_handle = workspace
        emode = workspace_handle.getEMode().name
        if emode == 'Elastic':
            # Work-around for older versions of Mantid which does not set instrument name
            # in NXSPE files, so LoadNXSPE does not know if it is direct or indirect data
            ei_log = workspace_handle.run().getProperty('Ei').value
            emode = 'Indirect' if np.isnan(ei_log) else 'Direct'
        return emode

    def has_efixed(self, workspace):
        return self._EfDefined[workspace if isinstance(workspace, basestring) else self.get_workspace_name(workspace)]

    def set_efixed(self, workspace, Ef):
        """Sets (overides) the fixed energy for all detectors (spectra) of this workspace"""
        ws_name = workspace if isinstance(workspace, basestring) else self.get_workspace_name(workspace)
        ws_handle = self.get_workspace_handle(ws_name)
        for idx in range(ws_handle.getNumberHistograms()):
            ws_handle.setEFixed(ws_handle.getDetector(idx).getID(), Ef)

    def get_limits(self, workspace, axis):
        """Determines the limits of the data and minimum step size"""
        if workspace not in self._limits.keys():
            self._processLoadedWSLimits(workspace)
        # If we cannot get the step size from the data, use the old 1/100 steps.
        if axis in self._limits[workspace].keys():
            return self._limits[workspace][axis]
        else:
            ws_h = self.get_workspace_handle(workspace)
            dim = ws_h.getDimension(ws_h.getDimensionIndexByName(axis))
            minimum = dim.getMinimum()
            maximum = dim.getMaximum()
            return minimum, maximum, (maximum - minimum)/100.

    def propagate_properties(self, old_workspace, new_workspace):
        """Propagates MSlice only properties of workspaces, e.g. limits"""
        if old_workspace in self._EfDefined.keys():
            self._EfDefined[new_workspace] = self._EfDefined[old_workspace]
        if old_workspace in self._limits.keys():
            self._limits[new_workspace] = self._limits[old_workspace]

    def getComment(self, workspace):
        if hasattr(workspace, 'getComment'):
            return workspace.getComment()
        ws_handle = self.get_workspace_handle(workspace)
        return ws_handle.getComment()
