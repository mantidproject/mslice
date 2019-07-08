"""
Uses mantid algorithms to perform workspace operations
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)

import os.path

import numpy as np
from scipy import constants
from six import string_types

from mantid.api import WorkspaceUnitValidator
from mantid.api import MatrixWorkspace

from mslice.models.axis import Axis

from mslice.util.mantid.algorithm_wrapper import add_to_ads
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, get_workspace_name
from mslice.util.mantid.mantid_algorithms import Load, MergeMD, MergeRuns, Scale, Minus, ConvertUnits
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.workspace.workspace import Workspace

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

# Defines some conversion factors
E2q = 2. * constants.m_n / (constants.hbar ** 2)  # Energy to (neutron momentum)^2 (==2m_n/hbar^2)
meV2J = constants.e / 1000.  # meV to Joules
m2A = 1.e10  # metres to Angstrom


def get_limits(workspace, axis):
    workspace = get_workspace_handle(workspace)
    if workspace.limits is None or len(workspace.limits) == 0:
        _processLoadedWSLimits(workspace)
    if axis in workspace.limits:
        return workspace.limits[axis]
    else:
        # If we cannot get the step size from the data, use the old 1/100 steps.
        ws_h = workspace.raw_ws
        dim = ws_h.getDimension(ws_h.getDimensionIndexByName(axis))
        minimum = dim.getMinimum()
        maximum = dim.getMaximum()
        step = (maximum - minimum) / 100
        return minimum, maximum, step


def processEfixed(workspace):
    """Checks whether the fixed energy is defined for this workspace"""
    try:
        workspace.e_fixed = _get_ws_EFixed(workspace.raw_ws)
        workspace.ef_defined = True
    except RuntimeError:
        workspace.ef_defined = False


def _processLoadedWSLimits(workspace):
    """ Processes an (angle-deltaE) workspace to get the limits and step size in angle, energy and |Q| """
    # For cases, e.g. indirect, where EFixed has not been set yet, return calculate later.
    if workspace.e_fixed is None:
        workspace.e_fixed = get_EFixed(workspace.raw_ws)
        if workspace.e_fixed is None:
            return
    if isinstance(workspace, PixelWorkspace):
        process_limits_event(workspace)
    elif isinstance(workspace, Workspace):
        process_limits(workspace)


def process_limits(ws):
    en = ws.raw_ws.getAxis(0).extractValues()
    theta = _get_theta_for_limits(ws)
    qmin, qmax, qstep = get_q_limits(theta, en, ws.e_fixed)
    set_limits(ws, qmin, qmax, qstep, theta, np.min(en), np.max(en), np.mean(np.diff(en)))


def process_limits_event(ws):
    e_dim = ws.raw_ws.getDimension(ws.raw_ws.getDimensionIndexByName('DeltaE'))
    emin = e_dim.getMinimum()
    emax = e_dim.getMaximum()
    theta = _get_theta_for_limits_event(ws)
    estep = _original_step_size(ws)
    emax_1 = -emin if (str(ws.e_mode == 'Direct')) else emax
    qmin, qmax, qstep = get_q_limits(theta, emax_1, ws.e_fixed)
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


def get_q_limits(theta, en, efix):
    #calculates the Q(E) line for the given two theta and then finds the min and max values
    qlines = [np.sqrt(E2q * (2 * efix - en - 2 * np.sqrt(efix * (efix - en)) * np.cos(tth)) * meV2J) / 1e10 for tth in theta[:2]]
    qmin = np.nanmin(qlines[0])
    qmax = np.nanmax(qlines[1])
    const_tth = np.radians(0.5)
    qstep = np.sqrt(E2q * 2 * efix * (1 - np.cos(const_tth)) * meV2J) / m2A
    qmin -= qstep
    qmax += qstep
    return qmin, qmax, qstep


def set_limits(ws, qmin, qmax, qstep, theta, emin, emax, estep):

    ws.limits['MomentumTransfer'] = [qmin, qmax, qstep]
    ws.limits['|Q|'] = ws.limits['MomentumTransfer']  # ConvertToMD renames it(!)
    ws.limits['2Theta'] = theta * 180 / np.pi
    ws.limits['DeltaE'] = [emin, emax, estep]


def _get_theta_for_limits(ws):
    num_hist = ws.raw_ws.getNumberHistograms()
    theta = [ws.raw_ws.detectorTwoTheta(ws.raw_ws.getDetector(i)) for i in range(num_hist)]
    round_fac = 100
    ws.is_PSD = not all(x < y for x, y in zip(theta, theta[1:]))
    # Rounds the differences to avoid pixels with same 2theta. Implies min limit of ~0.5 degrees
    thdiff = np.diff(np.round(np.sort(theta)*round_fac)/round_fac)
    return np.array([np.min(theta), np.max(theta), np.min(thdiff[np.where(thdiff > 0)])])


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
    workspace = Load(OutputWorkspace=output_workspace, Filename=filename)
    workspace.e_mode = get_EMode(workspace.raw_ws)
    if workspace.e_mode == 'Indirect':
        processEfixed(workspace)
    if (isinstance(workspace.raw_ws, MatrixWorkspace)
            and WorkspaceUnitValidator("DeltaE_inWavenumber").isValid(workspace.raw_ws)) == '':
        workspace = ConvertUnits(InputWorkspace=workspace, Target="DeltaE", EMode=workspace.e_mode,
                                 OutputWorkspace=workspace.name)
    _processLoadedWSLimits(workspace)
    return workspace


def combine_workspace(selected_workspaces, new_name):
    workspaces = [get_workspace_handle(ws) for ws in selected_workspaces]
    workspace_names = [workspace.name for workspace in workspaces]
    ws = MergeMD(OutputWorkspace=new_name, InputWorkspaces=workspace_names)
    propagate_properties(workspaces[0], ws)
    # Set limits for result workspace. Use precalculated step size, otherwise get limits directly from Mantid workspace
    ax1 = ws.raw_ws.getDimension(0)
    ax2 = ws.raw_ws.getDimension(1)
    step1 = []
    step2 = []
    for in_ws in selected_workspaces:
        step1.append(get_limits(in_ws, ax1.name)[2])
        step2.append(get_limits(in_ws, ax2.name)[2])
    ws.limits[ax1.name] = [ax1.getMinimum(), ax1.getMaximum(), np.max(step1)]
    ws.limits[ax2.name] = [ax2.getMinimum(), ax2.getMaximum(), np.max(step2)]
    return ws


def add_workspace_runs(selected_ws):
    out_ws_name = selected_ws[0] + '_sum'
    sum_ws = MergeRuns(OutputWorkspace=out_ws_name, InputWorkspaces=selected_ws)
    propagate_properties(get_workspace_handle(selected_ws[0]), sum_ws)


def subtract(workspaces, background_ws, ssf):
    scaled_bg_ws = Scale(OutputWorkspace=str(background_ws) + '_scaled',
                         InputWorkspace=str(background_ws), Factor=ssf, store=False)
    try:
        for ws_name in workspaces:
            result = Minus(OutputWorkspace=ws_name + '_subtracted', LHSWorkspace=ws_name,
                           RHSWorkspace=scaled_bg_ws)
            propagate_properties(get_workspace_handle(ws_name), result)
    except ValueError as e:
        raise ValueError(e)


def rebose_single(ws, from_temp, to_temp):
    from mslice.util.mantid.mantid_algorithms import Rebose
    ws = get_workspace_handle(ws)
    results = Rebose(InputWorkspace=ws, CurrentTemperature=from_temp, TargetTemperature=to_temp,
                     OutputWorkspace=ws.name+'_bosed')
    propagate_properties(ws, results)
    return results

def scale_workspaces(workspaces, scale_factor=None, from_temp=None, to_temp=None):
    if scale_factor is None:
        if from_temp is None or to_temp is None:
            raise ValueError('You must specify all inputs (scale factor or both temperatures)')
        for ws_name in workspaces:
            rebose_single(ws_name, from_temp, to_temp)
    else:
        for ws_name in workspaces:
            ws = get_workspace_handle(ws_name)
            result = Scale(InputWorkspace=ws.raw_ws, Factor=scale_factor, OutputWorkspace=ws_name+'_scaled')
            propagate_properties(ws, result)


def save_workspaces(workspaces, path, save_name, extension, slice_nonpsd=False):
    """
    :param workspaces: list of workspaces to save
    :param path: directory to save to
    :param save_name: name to save the file as (plus file extension). Pass none to use workspace name
    :param extension: file extension (such as .txt)
    :param slice_nonpsd: whether the selection is in non_psd mode
    """
    from .file_io import save_ascii, save_matlab, save_nexus
    if extension == '.nxs':
        save_method = save_nexus
    elif extension == '.txt':
        save_method = save_ascii
    elif extension == '.mat':
        save_method = save_matlab
    else:
        raise RuntimeError("unrecognised file extension")
    if isinstance(save_name, string_types):
        if len(workspaces) == 1:
            save_names = [save_name]
        else:
            from os.path import splitext
            name, _ = splitext(save_name)
            save_names = ['{}_{:03d}{}'.format(name, idx+1, extension) for idx in range(len(workspaces))]
    else:
        save_names = [save_name] if not hasattr(save_name, '__iter__') else save_name
        if len(save_names) != len(workspaces):
            save_names = [None] * len(workspaces)
    for workspace, save_name_single in zip(workspaces, save_names):
        _save_single_ws(workspace, save_name_single, save_method, path, extension, slice_nonpsd)


def export_workspace_to_ads(workspace):
    """
    Exports an MSlice workspace to ADS. If the workspace is MDHisto, convert it to Matrix
    :param workspace: name of MSlice workspace to export to ADS
    """
    workspace = get_workspace_handle(workspace)
    if isinstance(workspace, HistogramWorkspace):
        workspace = workspace.convert_to_matrix()
    add_to_ads(workspace)


def _save_single_ws(workspace, save_name, save_method, path, extension, slice_nonpsd):
    slice = False
    save_as = save_name if save_name is not None else str(workspace) + extension
    full_path = os.path.join(str(path), save_as)
    workspace = get_workspace_handle(workspace)
    non_psd_slice = slice_nonpsd and isinstance(workspace, Workspace) and not workspace.is_PSD
    if is_pixel_workspace(workspace) or non_psd_slice:
        slice = True
        workspace = _get_slice_mdhisto(workspace, get_workspace_name(workspace))
    save_method(workspace, full_path, slice)


def _get_slice_mdhisto(workspace, ws_name):
    from mslice.models.slice.slice_functions import compute_slice
    try:
        return get_workspace_handle('__' + ws_name)
    except KeyError:
        x_axis = get_axis_from_dimension(workspace, ws_name, 0)
        y_axis = get_axis_from_dimension(workspace, ws_name, 1)
        compute_slice(ws_name, x_axis, y_axis, False)
        return get_workspace_handle('__' + ws_name)


def get_axis_from_dimension(workspace, ws_name, id):
    dim = workspace.raw_ws.getDimension(id).getName()
    min, max, step = workspace.limits[dim]
    return Axis(dim, min, max, step)


def is_pixel_workspace(workspace_name):
    workspace = get_workspace_handle(workspace_name)
    return isinstance(workspace, PixelWorkspace)


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


def get_EFixed(raw_ws):
    efix = np.nan
    try:
        efix = _get_ws_EFixed(raw_ws)
    except RuntimeError:  # Efixed not defined
        # This could occur for malformed NXSPE without the instrument name set.
        # LoadNXSPE then sets EMode to 'Elastic' and getEFixed fails.
        try:
            if raw_ws.run().hasProperty('Ei'):
                efix = raw_ws.run().getProperty('Ei').value
        except AttributeError:
            if raw_ws.getExperimentInfo(0).run().hasProperty('Ei'):
                efix = raw_ws.getExperimentInfo(0).run().getProperty('Ei').value
    if efix is not None and not np.isnan(efix):  # error if none is passed to isnan
        return efix
    else:
        return None


def _get_ws_EFixed(raw_ws):
    try:
        efixed = raw_ws.getEFixed(raw_ws.getDetector(0).getID())
    except AttributeError: # workspace is not matrix workspace
        try:
            efixed = _get_exp_info_using(raw_ws, lambda e: raw_ws.getExperimentInfo(e).getEFixed(1))
        except ValueError:
            raise ValueError("Workspace contains different EFixed values")
    return efixed


def _get_exp_info_using(raw_ws, get_exp_info):
    """get data from MultipleExperimentInfo. Returns None if ExperimentInfo is not found"""
    prev = None
    for exp in range(raw_ws.getNumExperimentInfo()):
        exp_value = get_exp_info(exp)
        if prev is not None:
            if exp_value != prev:
                raise ValueError
        prev = exp_value
    return prev


def propagate_properties(old_workspace, new_workspace):
    """Propagates MSlice only properties of workspaces, e.g. limits"""
    new_workspace.ef_defined = old_workspace.ef_defined
    new_workspace.e_mode = old_workspace.e_mode
    new_workspace.limits = old_workspace.limits
    new_workspace.is_PSD = old_workspace.is_PSD
    new_workspace.e_fixed = old_workspace.e_fixed


def get_comment(workspace):
    if hasattr(workspace, 'getComment'):
        return workspace.getComment()
    if not hasattr(workspace, 'raw_ws'):
        workspace = get_workspace_handle(workspace)
    return workspace.raw_ws.getComment()
