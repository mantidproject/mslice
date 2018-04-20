"""A concrete implementation of a WorkspaceProvider

It uses mantid to perform the workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)
import os.path
from six import string_types
from mantid.api import IMDEventWorkspace, IMDHistoWorkspace

from mantid.simpleapi import (DeleteWorkspace, Load, Scale, RenameWorkspace, 
                              MergeMD, MergeRuns, Minus)

from mslice.workspace.base import WorkspaceBase as Workspace
from mslice.workspace.workspace import Workspace as MatrixWorkspace
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.histogram_workspace import HistogramWorkspace
from .file_io import save_ascii, save_matlab, save_nexus
import numpy as np
from scipy import constants

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

# Defines some conversion factors
E2q = 2. * constants.m_n / (constants.hbar ** 2)  # Energy to (neutron momentum)^2 (==2m_n/hbar^2)
meV2J = constants.e / 1000.  # meV to Joules
m2A = 1.e10  # metres to Angstrom

_loaded_workspaces = {}

class Axis(object):
    def __init__(self, units, start, end, step):
        self.units = units
        self.start = start
        self.end = end
        self.step = step

    def __eq__(self, other):
        # This is required for Unit testing
        return self.units == other.units and self.start == other.start and self.end == other.end \
            and self.step == other.step and isinstance(other, Axis)

    def __repr__(self):
        info = (self.units, self.start, self.end, self.step)
        return "Axis(" + " ,".join(map(repr, info)) + ")"

def get_workspace_handle(workspace_name):
    """"Return handle to workspace given workspace_name_as_string"""
    # if passed a workspace handle return the handle
    if isinstance(workspace_name, Workspace):
        return workspace_name
    return _loaded_workspaces[workspace_name]

def get_workspace_names():
    return _loaded_workspaces.keys()

def delete_workspace(workspace):
    workspace = get_workspace_handle(workspace)
    del _loaded_workspaces[get_workspace_name(workspace)]
    del workspace

def get_limits(workspace, axis):
    if workspace.limits is None:
        _processLoadedWSLimits(workspace)
    if axis in workspace.limits:
        return workspace.limits[axis]
    else:
        # If we cannot get the step size from the data, use the old 1/100 steps.
        ws_h = get_workspace_handle(workspace)
        dim = ws_h.getDimension(ws_h.getDimensionIndexByName(axis))
        minimum = dim.getMinimum()
        maximum = dim.getMaximum()
        step = (maximum - minimum) / 100
        return minimum, maximum, step

def is_PSD(workspace):
    ws_name = workspace if isinstance(workspace, string_types) else get_workspace_name(workspace)
    return _isPSD[ws_name] if (ws_name in _isPSD) else None

def _processEfixed(workspace):
    """Checks whether the fixed energy is defined for this workspace"""
    try:
        _get_ws_EFixed(workspace.raw_ws)
        workspace.ef_defined = True
    except RuntimeError:
        workspace.ef_defined = False

def _processLoadedWSLimits(workspace):
    """ Processes an (angle-deltaE) workspace to get the limits and step size in angle, energy and |Q| """
    # For cases, e.g. indirect, where EFixed has not been set yet, return calculate later.
    efix = get_EFixed(workspace.raw_ws)
    if efix is None:
        return
    if isinstance(workspace, PixelWorkspace):
        process_limits_event(workspace, efix)
    elif isinstance(workspace, MatrixWorkspace):
        process_limits(workspace, efix)

def process_limits(ws, efix):
    en = ws.raw_ws.getAxis(0).extractValues()
    theta = _get_theta_for_limits(ws)
    # Use minimum energy (Direct geometry) or maximum energy (Indirect) to get qmax
    emax = -np.min(en) if (str(ws.e_mode == 'Direct')) else np.max(en)
    qmin, qmax, qstep = get_q_limits(theta, emax, efix)
    set_limits(ws, qmin, qmax, qstep, theta, np.min(en), np.max(en), np.mean(np.diff(en)))

def process_limits_event(ws, efix):
    e_dim = ws.raw_ws.getDimension(ws.raw_ws.getDimensionIndexByName('DeltaE'))
    emin  = e_dim.getMinimum()
    emax = e_dim.getMaximum()
    theta = _get_theta_for_limits_event(ws)
    estep = _original_step_size(ws)
    emax_1 = -emin if (str(ws.e_mode == 'Direct')) else emax
    qmin, qmax, qstep = get_q_limits(theta, emax_1, efix)
    set_limits(ws, qmin, qmax, qstep, theta, emin, emax, estep)

def _original_step_size(workspace):
    rebin_history = _get_algorithm_history("Rebin", workspace.raw_ws.getHistory())
    params_history = _get_property_from_history("Params", rebin_history)
    return float(params_history.value().split(',')[1])

def _get_algorithm_history(name, workspace_history):
    histories = workspace_history.getAlgorithmHistories()

    for history in reversed(histories):
        if history.name() == name:
            return history
    return None

def _get_property_from_history(name, history):
    for property in history.getProperties():
        if property.name() == name:
            return property
    return None

def get_q_limits(theta, emax, efix):
    qmin, qmax, qstep = tuple(np.sqrt(E2q * 2 * efix * (1 - np.cos(theta)) * meV2J) / m2A)
    qmax = np.sqrt(E2q * (2 * efix + emax - 2 * np.sqrt(efix * (efix + emax)) * np.cos(theta[1])) * meV2J) / m2A
    return qmin, qmax, qstep

def set_limits(ws, qmin, qmax, qstep, theta, emin, emax, estep):
    # Use a step size a bit smaller than angular spacing ( / 3) so user can rebin if they want...
    ws.limits['MomentumTransfer'] = [qmin - qstep, qmax + qstep, qstep / 3]
    ws.limits['|Q|'] = ws.limits['MomentumTransfer']  # ConvertToMD renames it(!)
    ws.limits['Degrees'] = theta * 180 / np.pi
    ws.limits['DeltaE'] = [emin, emax, estep]

def _get_theta_for_limits(ws):
    # Don't parse all spectra in cases where there are a lot to save time.
    num_hist = ws.raw_ws.getNumberHistograms()
    if num_hist > 1000:
        n_segments = 5
        interval = int(num_hist / n_segments)
        theta = []
        for segment in range(n_segments):
            i0 = segment * interval
            theta.append([ws.raw_ws.detectorTwoTheta(ws.raw_ws.getDetector(i))
                          for i in range(i0, i0+200)])
        round_fac = 573
    else:
        theta = [ws.raw_ws.detectorTwoTheta(ws.raw_ws.getDetector(i)) for i in range(num_hist)]
        round_fac = 100
    ws.is_PSD = not all(x < y for x, y in zip(theta, theta[1:]))
    # Rounds the differences to avoid pixels with same 2theta. Implies min limit of ~0.5 degrees
    thdiff = np.diff(np.round(np.sort(theta)*round_fac)/round_fac)
    return np.array([np.min(theta), np.max(theta), np.min(thdiff[np.where(thdiff>0)])])

def _get_theta_for_limits_event(ws):
    spectrum_info = ws.raw_ws.getExperimentInfo(0).spectrumInfo()
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

def load(filename, output_workspace):
    ws = Load(Filename=filename, OutputWorkspace=output_workspace)
    wrapped = wrap_workspace(ws)
    wrapped.e_mode = get_EMode(ws)
    if wrapped.e_mode == 'Indirect':
        _processEfixed(wrapped)
    _processLoadedWSLimits(wrapped)
    return wrapped

def wrap_workspace(raw_ws):
    if isinstance(raw_ws, IMDEventWorkspace):
        wrapped = PixelWorkspace(raw_ws)
    elif isinstance(raw_ws, IMDHistoWorkspace):
        wrapped = HistogramWorkspace(raw_ws)
    else:
        wrapped = MatrixWorkspace(raw_ws)
    _loaded_workspaces[raw_ws.name()] = wrapped
    return wrapped


def rename_workspace(selected_workspace, new_name):
    ws = RenameWorkspace(InputWorkspace=selected_workspace, OutputWorkspace=new_name)
    if selected_workspace in _limits:
        _limits[new_name] = _limits.pop(selected_workspace)
    if selected_workspace in _isPSD:
        _isPSD[new_name] = _isPSD.pop(selected_workspace)
    if selected_workspace in _EfDefined:
        _EfDefined[new_name] = _EfDefined.pop(selected_workspace)
    if selected_workspace in _cutParameters:
        _cutParameters[new_name] = _cutParameters.pop(selected_workspace)
    return ws

def combine_workspace(selected_workspaces, new_name):
    ws = MergeMD(InputWorkspaces=selected_workspaces, OutputWorkspace=new_name)
    # Use precalculated step size, otherwise get limits directly from workspace
    ax1 = ws.getDimension(0)
    ax2 = ws.getDimension(1)
    step1 = []
    step2 = []
    for input_workspace in selected_workspaces:
        step1.append(get_limits(input_workspace, ax1.name)[2])
        step2.append(get_limits(input_workspace, ax2.name)[2])
    if new_name not in _limits.keys():
        _limits[new_name] = {}
    _limits[new_name][ax1.name] = [ax1.getMinimum(), ax1.getMaximum(), np.max(step1)]
    _limits[new_name][ax2.name] = [ax2.getMinimum(), ax2.getMaximum(), np.max(step2)]
    return ws

def add_workspace_runs(selected_ws):
    MergeRuns(InputWorkspaces=selected_ws, OutputWorkspace=selected_ws[0] + '_sum')

def subtract(workspaces, background_ws, ssf):
    bg_ws = get_workspace_handle(str(background_ws))
    scaled_bg_ws = Scale(bg_ws, ssf)
    try:
        for ws_name in workspaces:
            ws = get_workspace_handle(ws_name)
            Minus(LHSWorkspace=ws, RHSWorkspace=scaled_bg_ws, OutputWorkspace=ws_name + '_subtracted')
    except ValueError as e:
        raise ValueError(e)
    finally:
        delete_workspace(scaled_bg_ws)

def save_workspaces(workspaces, path, save_name, extension, slice_nonpsd=False):
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
        _save_single_ws(workspace, save_name, save_method, path, extension, slice_nonpsd)

def _save_single_ws(workspace, save_name, save_method, path, extension, slice_nonpsd):
    slice = False
    save_as = save_name if save_name is not None else str(workspace) + extension
    full_path = os.path.join(str(path), save_as)
    workspace = get_workspace_handle(workspace)
    non_psd_slice = slice_nonpsd and not is_PSD(workspace) and isinstance(workspace, MatrixWorkspace)
    if is_pixel_workspace(workspace) or non_psd_slice:
        slice = True
        workspace = _get_slice_mdhisto(workspace, workspace.name())
    save_method(workspace, full_path, slice)

def _get_slice_mdhisto(workspace, ws_name):
    from mslice.models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
    try:
        return get_workspace_handle('__' + ws_name)
    except KeyError:
        slice_alg = MantidSliceAlgorithm()
        ws_name = workspace.name()
        x_axis = get_axis_from_dimension(workspace, ws_name, 0)
        y_axis = get_axis_from_dimension(workspace, ws_name, 1)
        slice_alg.compute_slice(ws_name, x_axis, y_axis, False)
        return get_workspace_handle('__' + ws_name)

def get_axis_from_dimension(workspace, ws_name, id):
    dim = workspace.getDimension(id).getName()
    min, max, step = _limits[ws_name][dim]
    return Axis(dim, min, max, step)


def is_pixel_workspace(workspace_name):
    workspace = get_workspace_handle(workspace_name)
    return isinstance(workspace, IMDEventWorkspace)

def get_workspace_name(workspace):
    """Returns the name of a workspace given the workspace handle"""
    if isinstance(workspace, string_types):
        return workspace
    return workspace.raw_ws.name()

def get_EMode(workspace):
    """Returns the energy analysis mode (direct or indirect of a workspace)"""
    emode = str(_get_ws_EMode(workspace))
    if emode == 'Elastic':
        # Work-around for older versions of Mantid which does not set instrument name
        # in NXSPE files, so LoadNXSPE does not know if it is direct or indirect data
        ei_log = workspace.run().getProperty('Ei').value
        emode = 'Indirect' if np.isnan(ei_log) else 'Direct'
    return emode

def _get_ws_EMode(ws_handle):
    try:
        emode = ws_handle.getEMode()
    except AttributeError: # workspace is not matrix workspace
        try:
            emode = _get_exp_info_using(ws_handle, lambda e: ws_handle.getExperimentInfo(e).getEMode())
        except ValueError:
            raise ValueError("Workspace contains different EModes")
    return emode

def get_EFixed(ws_handle):
    efix = np.nan
    try:
        efix = _get_ws_EFixed(ws_handle)
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

def _get_ws_EFixed(ws_handle):
    try:
        efixed = ws_handle.getEFixed(ws_handle.getDetector(0).getID())
    except AttributeError: # workspace is not matrix workspace
        try:
            efixed = _get_exp_info_using(ws_handle, lambda e: ws_handle.getExperimentInfo(e).getEFixed(1))
        except ValueError:
            raise ValueError("Workspace contains different EFixed values")
    return efixed

def _get_exp_info_using(ws_handle, get_exp_info):
    """get data from MultipleExperimentInfo. Returns None if ExperimentInfo is not found"""
    prev = None
    for exp in range(ws_handle.getNumExperimentInfo()):
        exp_value = get_exp_info(exp)
        if prev is not None:
            if exp_value != prev:
                raise ValueError
        prev = exp_value
    return prev

def has_efixed(workspace):
    return _EfDefined[workspace if isinstance(workspace, string_types) else get_workspace_name(workspace)]

def set_efixed(workspace, Ef):
    """Sets (overides) the fixed energy for all detectors (spectra) of this workspace"""
    ws_name = workspace if isinstance(workspace, string_types) else get_workspace_name(workspace)
    ws_handle = get_workspace_handle(ws_name)
    for idx in range(ws_handle.getNumberHistograms()):
        ws_handle.setEFixed(ws_handle.getDetector(idx).getID(), Ef)

def propagate_properties(old_workspace, new_workspace):
    """Propagates MSlice only properties of workspaces, e.g. limits"""
    new_ws = wrap_workspace(new_workspace)
    new_ws.ef_defined = old_workspace.ef_defined
    new_ws.e_mode = old_workspace.e_mode
    new_ws.limits = old_workspace.limits
    new_ws.is_PSD = old_workspace.is_PSD
    return new_ws

def getComment(workspace):
    if hasattr(workspace, 'getComment'):
        return workspace.getComment()
    ws_handle = get_workspace_handle(workspace)
    return ws_handle.raw_ws.getComment()
