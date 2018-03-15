"""A concrete implementation of a WorkspaceProvider

It uses mantid to perform the workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)
import os.path
from six import string_types

from mantid.simpleapi import (AnalysisDataService, DeleteWorkspace, Load, Scale,
                              RenameWorkspace, MergeMD, MergeRuns, Minus)

from mslice.presenters.slice_plotter_presenter import Axis
from mantid.api import IMDEventWorkspace, MatrixWorkspace, Workspace
from .file_io import save_ascii, save_matlab, save_nexus
import numpy as np
from scipy import constants

from .workspace_provider import WorkspaceProvider

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

# Defines some conversion factors
E2q = 2. * constants.m_n / (constants.hbar ** 2)  # Energy to (neutron momentum)^2 (==2m_n/hbar^2)
meV2J = constants.e / 1000.  # meV to Joules
m2A = 1.e10  # metres to Angstrom


class MantidWorkspaceProvider(WorkspaceProvider):
    def __init__(self):
        # Stores various parameters of workspaces not stored by Mantid
        self._EfDefined = {}
        self._limits = {}
        self._cutParameters = {}
        self._isPSD = {}

    def get_workspace_handle(self, workspace_name):
        """"Return handle to workspace given workspace_name_as_string"""
        # if passed a workspace handle return the handle
        if isinstance(workspace_name, Workspace):
            return workspace_name
        return AnalysisDataService[str(workspace_name)]

    def get_workspace_names(self):
        return AnalysisDataService.getObjectNames()

    def delete_workspace(self, workspace):
        ws = DeleteWorkspace(Workspace=workspace)
        if workspace in self._EfDefined:
            del self._EfDefined[workspace]
        if workspace in self._limits:
            del self._limits[workspace]
        if workspace in self._isPSD:
            del self._isPSD[workspace]
        return ws

    def get_limits(self, workspace, axis):
        if workspace not in self._limits:
            self._processLoadedWSLimits(workspace)
        if axis in self._limits[workspace]:
            return self._limits[workspace][axis]
        else:
            # If we cannot get the step size from the data, use the old 1/100 steps.
            ws_h = self.get_workspace_handle(workspace)
            dim = ws_h.getDimension(ws_h.getDimensionIndexByName(axis))
            minimum = dim.getMinimum()
            maximum = dim.getMaximum()
            step = (maximum - minimum) / 100
            return minimum, maximum, step

    def is_PSD(self, workspace):
        ws_name = workspace if isinstance(workspace, string_types) else self.get_workspace_name(workspace)
        return self._isPSD[ws_name] if (ws_name in self._isPSD) else None

    def _processEfixed(self, workspace):
        """Checks whether the fixed energy is defined for this workspace"""
        ws_name = workspace if isinstance(workspace, string_types) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(ws_name)
        try:
            self._get_ws_EFixed(ws_h)
            self._EfDefined[ws_name] = True
        except RuntimeError:
            self._EfDefined[ws_name] = False

    def _processLoadedWSLimits(self, workspace):
        """ Processes an (angle-deltaE) workspace to get the limits and step size in angle, energy and |Q| """
        ws_name = workspace if isinstance(workspace, string_types) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(workspace)
        # For cases, e.g. indirect, where EFixed has not been set yet, return calculate later.
        efix = self.get_EFixed(ws_h)
        if efix is None:
            return
        if ws_name not in self._limits:
            self._limits[ws_name] = {}
        if isinstance(ws_h, IMDEventWorkspace):
            self.process_limits_event(ws_h, ws_name, efix)
        elif isinstance(ws_h, MatrixWorkspace):
            self.process_limits(ws_h, ws_name, efix)

    def process_limits(self, ws, ws_name, efix):
        en = ws.getAxis(0).extractValues()
        theta = self._get_theta_for_limits(ws)
        # Use minimum energy (Direct geometry) or maximum energy (Indirect) to get qmax
        emax = -np.min(en) if (str(ws.getEMode()) == 'Direct') else np.max(en)
        qmin, qmax, qstep = self.get_q_limits(theta, emax, efix)
        self.set_limits(ws_name, qmin, qmax, qstep, theta, np.min(en), np.max(en), np.mean(np.diff(en)))

    def process_limits_event(self, ws, ws_name, efix):
        e_dim = ws.getDimension(ws.getDimensionIndexByName('DeltaE'))
        emin  = e_dim.getMinimum()
        emax = e_dim.getMaximum()
        theta = self._get_theta_for_limits_event(ws)
        estep = self._original_step_size(ws)
        emax_1 = -emin if (str(self.get_EMode(ws)) == 'Direct') else emax
        qmin, qmax, qstep = self.get_q_limits(theta, emax_1, efix)
        self.set_limits(ws_name, qmin, qmax, qstep, theta, emin, emax, estep)

    def _original_step_size(self, workspace):
        rebin_history = self._get_algorithm_history("Rebin", workspace.getHistory())
        params_history = self._get_property_from_history("Params", rebin_history)
        return float(params_history.value().split(',')[1])

    def _get_algorithm_history(self, name, workspace_history):
        histories = workspace_history.getAlgorithmHistories()

        for history in reversed(histories):
            if history.name() == name:
                return history
        return None

    def _get_property_from_history(self, name, history):
        for property in history.getProperties():
            if property.name() == name:
                return property
        return None

    def get_q_limits(self, theta, emax, efix):
        qmin, qmax, qstep = tuple(np.sqrt(E2q * 2 * efix * (1 - np.cos(theta)) * meV2J) / m2A)
        qmax = np.sqrt(E2q * (2 * efix + emax - 2 * np.sqrt(efix * (efix + emax)) * np.cos(theta[1])) * meV2J) / m2A
        return qmin, qmax, qstep

    def set_limits(self, ws_name, qmin, qmax, qstep, theta, emin, emax, estep):
        # Use a step size a bit smaller than angular spacing ( / 3) so user can rebin if they want...
        self._limits[ws_name]['MomentumTransfer'] = [qmin - qstep, qmax + qstep, qstep / 3]
        self._limits[ws_name]['|Q|'] = self._limits[ws_name]['MomentumTransfer']  # ConvertToMD renames it(!)
        self._limits[ws_name]['Degrees'] = theta * 180 / np.pi
        self._limits[ws_name]['DeltaE'] = [emin, emax, estep]

    def _get_theta_for_limits(self, ws_handle):
        # Don't parse all spectra in cases where there are a lot to save time.
        num_hist = ws_handle.getNumberHistograms()
        if num_hist > 1000:
            n_segments = 5
            interval = int(num_hist / n_segments)
            theta = []
            for segment in range(n_segments):
                i0 = segment * interval
                theta.append([ws_handle.detectorTwoTheta(ws_handle.getDetector(i))
                              for i in range(i0, i0+200)])
            round_fac = 573
        else:
            theta = [ws_handle.detectorTwoTheta(ws_handle.getDetector(i)) for i in range(num_hist)]
            round_fac = 100
        self._isPSD[self.get_workspace_name(ws_handle)] = not all(x < y for x, y in zip(theta, theta[1:]))
        # Rounds the differences to avoid pixels with same 2theta. Implies min limit of ~0.5 degrees
        thdiff = np.diff(np.round(np.sort(theta)*round_fac)/round_fac)
        return np.array([np.min(theta), np.max(theta), np.min(thdiff[np.where(thdiff>0)])])

    def _get_theta_for_limits_event(self, ws):
        spectrum_info = ws.getExperimentInfo(0).spectrumInfo()
        theta = []
        i = 0
        while True:
            try:
                if not spectrum_info.isMonitor(i):
                    theta.append(spectrum_info.twoTheta(i))
                i += 1
            except IndexError:
                break
        theta = np.unique(theta)
        round_fac = 100
        thdiff = np.diff(np.round(np.sort(theta) * round_fac) / round_fac)
        return np.array([np.min(theta), np.max(theta), np.min(thdiff[np.where(thdiff > 0)])])

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
        if selected_workspace in self._isPSD:
            self._isPSD[new_name] = self._isPSD.pop(selected_workspace)
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

    def add_workspace_runs(self, selected_ws):
        MergeRuns(InputWorkspaces=selected_ws, OutputWorkspace=selected_ws[0] + '_sum')

    def subtract(self, workspaces, background_ws, ssf):
        bg_ws = self.get_workspace_handle(str(background_ws))
        scaled_bg_ws = Scale(bg_ws, ssf)
        try:
            for ws_name in workspaces:
                ws = self.get_workspace_handle(ws_name)
                Minus(LHSWorkspace=ws, RHSWorkspace=scaled_bg_ws, OutputWorkspace=ws_name + '_subtracted')
        except ValueError as e:
            raise ValueError(e)
        finally:
            self.delete_workspace(scaled_bg_ws)

    def save_workspaces(self, workspaces, path, save_name, extension, slice_nonpsd=False):
        '''
        :param workspaces: list of workspaces to save
        :param path: directory to save to
        :param save_name: name to save the file as (plus file extension). Pass none to use workspace name
        :param extension: file extension (such as .txt)
        '''
        if extension == '.nxs':
            save_method = save_nexus
        elif extension == '.txt':
            save_method = save_ascii
        elif extension == '.mat':
            save_method = save_matlab
        else:
            raise RuntimeError("unrecognised file extension")
        for workspace in workspaces:
            self._save_single_ws(workspace, save_name, save_method, path, extension, slice_nonpsd)

    def _save_single_ws(self, workspace, save_name, save_method, path, extension, slice_nonpsd):
        slice = False
        save_as = save_name if save_name is not None else str(workspace) + extension
        full_path = os.path.join(str(path), save_as)
        workspace = self.get_workspace_handle(workspace)
        non_psd_slice = slice_nonpsd and not self.is_PSD(workspace) and isinstance(workspace, MatrixWorkspace)
        if self.is_pixel_workspace(workspace) or non_psd_slice:
            slice = True
            workspace = self._get_slice_mdhisto(workspace, workspace.name())
        save_method(workspace, full_path, slice)

    def _get_slice_mdhisto(self, workspace, ws_name):
        from mslice.models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
        try:
            return self.get_workspace_handle('__' + ws_name)
        except KeyError:
            slice_alg = MantidSliceAlgorithm()
            slice_alg.set_workspace_provider(self)
            ws_name = workspace.name()
            x_axis = self.get_axis_from_dimension(workspace, ws_name, 0)
            y_axis = self.get_axis_from_dimension(workspace, ws_name, 1)
            slice_alg.compute_slice(ws_name, x_axis, y_axis, False)
            return self.get_workspace_handle('__' + ws_name)

    def get_axis_from_dimension(self, workspace, ws_name, id):
        dim = workspace.getDimension(id).getName()
        min, max, step = self._limits[ws_name][dim]
        return Axis(dim, min, max, step)


    def is_pixel_workspace(self, workspace_name):
        workspace = self.get_workspace_handle(workspace_name)
        return isinstance(workspace, IMDEventWorkspace)

    def get_workspace_name(self, workspace):
        """Returns the name of a workspace given the workspace handle"""
        if isinstance(workspace, string_types):
            return workspace
        return workspace.name()

    def get_EMode(self, workspace):
        """Returns the energy analysis mode (direct or indirect of a workspace)"""
        workspace_handle = self.get_workspace_handle(workspace)
        emode = str(self._get_ws_EMode(workspace_handle))
        if emode == 'Elastic':
            # Work-around for older versions of Mantid which does not set instrument name
            # in NXSPE files, so LoadNXSPE does not know if it is direct or indirect data
            ei_log = workspace_handle.run().getProperty('Ei').value
            emode = 'Indirect' if np.isnan(ei_log) else 'Direct'
        return emode

    def _get_ws_EMode(self, ws_handle):
        try:
            emode = ws_handle.getEMode()
        except AttributeError: # workspace is not matrix workspace
            try:
                emode = self._get_exp_info_using(ws_handle, lambda e: ws_handle.getExperimentInfo(e).getEMode())
            except ValueError:
                raise ValueError("Workspace contains different EModes")
        return emode

    def get_EFixed(self, ws_handle):
        efix = np.nan
        try:
            efix = self._get_ws_EFixed(ws_handle)
        except RuntimeError:  # Efixed not defined
            # This could occur for malformed NXSPE without the instrument name set.
            # LoadNXSPE then sets EMode to 'Elastic' and getEFixed fails.
            try:
                if ws_handle.run().hasProperty('Ei'):
                    efix = ws_handle.run().getProperty('Ei').value
            except AttributeError:
                if ws_handle.getExperimentInfo(0).run().hasProperty('Ei'):
                    efix = ws_handle.getExperimentInfo(0).run().getProperty('Ei').value
        if efix is not None and not np.isnan(efix):  # error if none is passed to isnan
            return efix
        else:
            return None

    def _get_ws_EFixed(self, ws_handle):
        try:
            efixed = ws_handle.getEFixed(ws_handle.getDetector(0).getID())
        except AttributeError: # workspace is not matrix workspace
            try:
                efixed = self._get_exp_info_using(ws_handle, lambda e: ws_handle.getExperimentInfo(e).getEFixed(1))
            except ValueError:
                raise ValueError("Workspace contains different EFixed values")
        return efixed

    def _get_exp_info_using(self, ws_handle, get_exp_info):
        """get data from MultipleExperimentInfo. Returns None if ExperimentInfo is not found"""
        prev = None
        for exp in range(ws_handle.getNumExperimentInfo()):
            exp_value = get_exp_info(exp)
            if prev is not None:
                if exp_value != prev:
                    raise ValueError
            prev = exp_value
        return prev

    def has_efixed(self, workspace):
        return self._EfDefined[workspace if isinstance(workspace, string_types) else self.get_workspace_name(workspace)]

    def set_efixed(self, workspace, Ef):
        """Sets (overides) the fixed energy for all detectors (spectra) of this workspace"""
        ws_name = workspace if isinstance(workspace, string_types) else self.get_workspace_name(workspace)
        ws_handle = self.get_workspace_handle(ws_name)
        for idx in range(ws_handle.getNumberHistograms()):
            ws_handle.setEFixed(ws_handle.getDetector(idx).getID(), Ef)

    def propagate_properties(self, old_workspace, new_workspace):
        """Propagates MSlice only properties of workspaces, e.g. limits"""
        if old_workspace in self._EfDefined:
            self._EfDefined[new_workspace] = self._EfDefined[old_workspace]
        if old_workspace in self._limits:
            self._limits[new_workspace] = self._limits[old_workspace]
        if old_workspace in self._isPSD:
            self._isPSD[new_workspace] = self._isPSD[old_workspace]

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
