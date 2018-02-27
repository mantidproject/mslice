"""A concrete implementation of a WorkspaceProvider

It uses mantid to perform the workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)
import os.path
from six import string_types

from mantid.simpleapi import (AnalysisDataService, CreateMDHistoWorkspace, DeleteWorkspace, Load, Scale, SaveAscii,
                              RenameWorkspace, SaveNexus, SaveMD, MergeMD, MergeRuns, Minus)

from mantid.api import IMDEventWorkspace, IMDHistoWorkspace, Workspace
import numpy as np
from scipy import constants
from scipy.io import savemat

from .workspace_provider import WorkspaceProvider
from mslice.models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter
from mslice.models.slice.mantid_slice_algorithm import MantidSliceAlgorithm

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
        ws_name = workspace if isinstance(workspace, string_types) else self.get_workspace_name(workspace)
        ws_h = self.get_workspace_handle(ws_name)
        try:
            [self._get_ws_EFixed(ws_h, ws_h.getDetector(i).getID()) for i in range(ws_h.getNumberHistograms())]
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

    def get_EFixed(self, ws_handle):
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
        # Rounds the differences to avoid pixels with same 2theta. Implies min limit of ~0.5 degrees
        thdiff = np.diff(np.round(np.sort(theta)*round_fac)/round_fac)
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

    def save_workspace(self, workspaces, path, save_name, extension):
        '''
        :param workspaces: list of workspaces to save
        :param path: directory to save to
        :param save_name: name to save the file as (plus file extension). Pass none to use workspace name
        :param extension: file extension (such as .txt)
        '''
        if extension == '.nxs':
            save_method = self._save_nexus
        elif extension == '.txt':
            save_method = self._save_ascii
        elif extension == '.mat':
            save_method = self._save_matlab
        else:
            raise RuntimeError("unrecognised file extension")
        for workspace in workspaces:
            save_as = save_name if save_name is not None else str(workspace) + extension
            full_path = os.path.join(str(path), save_as)
            save_method(workspace, full_path)

    def _save_nexus(self, workspace, path):
        workspace_handle = self.get_workspace_handle(workspace)
        if isinstance(workspace_handle, IMDEventWorkspace) or isinstance(workspace_handle, IMDHistoWorkspace):
            SaveMD(InputWorkspace=workspace, Filename=path)
        else:
            SaveNexus(InputWorkspace=workspace, Filename=path)

    def _save_ascii(self, workspace, path):
        workspace_handle = self.get_workspace_handle(workspace)
        if isinstance(workspace_handle, IMDEventWorkspace):
            workspace = self.get_slice_MDHisto(workspace)
            workspace_handle = self.get_workspace_handle(workspace)
        elif isinstance(workspace_handle, IMDHistoWorkspace):
            self._save_cut_to_ascii(workspace_handle, workspace, path)
        else:
            SaveAscii(InputWorkspace=workspace, Filename=path)

    def _save_cut_to_ascii(self, workspace, ws_name, output_path):
        # get integration ranges from the name
        int_ranges = ws_name[ws_name.find('('):]
        int_start = int_ranges[1:int_ranges.find(',')]
        int_end = int_ranges[int_ranges.find(',')+1:-1]
        ws_name = ws_name[:ws_name.find('_cut')]

        dim = workspace.getDimension(0)
        units = dim.getUnits()

        x, y, e = self.get_md_histo_xye(workspace)
        header = 'MSlice Cut of workspace "%s" along "%s" between %s and %s' % (ws_name, units, int_start, int_end)
        out_data = np.c_[x, y, e]
        np.savetxt(str(output_path), out_data, fmt='%12.9e', header=header)

    def _save_matlab(self, workspace, path):
        workspace_handle = self.get_workspace_handle(workspace)
        if isinstance(workspace_handle, IMDEventWorkspace):
            workspace = self.get_slice_MDHisto(workspace)
            workspace_handle = self.get_workspace_handle(workspace)
        elif isinstance(workspace_handle, IMDHistoWorkspace):
            x, y, e = self.get_md_histo_xye(workspace_handle)
        else:
            x = workspace_handle.extractX()
            y = workspace_handle.extractY()
            e = workspace_handle.extractE()
        mdict = {'x': x, 'y': y, 'e': e}
        savemat(path, mdict=mdict)

    def load_from_ascii(self, file_path, ws_name):
        file = open(file_path, 'r')
        header = file.readline()
        if not header.startswith("# MSlice Cut"):
            raise ValueError
        x, y, e = np.loadtxt(file).transpose()
        extents = str(np.min(x)) + ',' + str(np.max(x))
        nbins = len(x)
        units = header[header.find('along "'):header.find('" between')]
        CreateMDHistoWorkspace(SignalInput=y, ErrorInput=e, Dimensionality=1, Extents=extents, NumberOfBins=nbins,
                               Names='Dim1', Units=units, OutputWorkspace=ws_name)

    def get_slice_MDHisto(self, workspace):
        try:
            self.get_workspace_handle('__' + workspace)
        except KeyError:
            slice_alg = MantidSliceAlgorithm()
            slice_alg.compute_slice()
            # need parameters
            # insert magic from ws_limits here once defaults can be obtained from slice

    def get_md_histo_xye(self, histo_ws):
        dim = histo_ws.getDimension(0)
        start = dim.getMinimum()
        end = dim.getMaximum()
        step = dim.getBinWidth()
        x = np.arange(start, end, step)
        y = histo_ws.getSignalArray()
        e = np.sqrt(histo_ws.getErrorSquaredArray())
        return x, y, e

    def is_pixel_workspace(self, workspace_name):
        from mantid.api import IMDEventWorkspace
        workspace = self.get_workspace_handle(workspace_name)
        return isinstance(workspace, IMDEventWorkspace)

    def get_workspace_handle(self, workspace_name):
        """"Return handle to workspace given workspace_name_as_string"""
        # if passed a workspace handle return the handle
        if isinstance(workspace_name, Workspace):
            return workspace_name
        return AnalysisDataService[str(workspace_name)]

    def get_parent_by_name(self, ws_name):
        if not isinstance(ws_name, string_types):
            ws_name = str(ws_name)
        suffixes = ('_QE', '_EQ', '_ETh', '_ThE')
        if ws_name.endswith(suffixes):
            return self.get_workspace_handle(ws_name.rsplit('_', 1)[0])
        else:
            return self.get_workspace_handle(ws_name)

    def get_workspace_name(self, workspace):
        """Returns the name of a workspace given the workspace handle"""
        if isinstance(workspace, string_types):
            return workspace
        return workspace.name()

    def get_EMode(self, workspace):
        """Returns the energy analysis mode (direct or indirect of a workspace)"""
        if isinstance(workspace, string_types):
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
