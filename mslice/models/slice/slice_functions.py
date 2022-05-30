from __future__ import (absolute_import, division, print_function)
from six import string_types
import numpy as np
from functools import partial

from mantid.api import WorkspaceUnitValidator
from scipy import constants

from mslice.models.alg_workspace_ops import get_number_of_steps
from mslice.models.labels import is_momentum, is_twotheta
from mslice.models.units import get_sample_temperature_from_string
from mslice.models.workspacemanager.workspace_algorithms import propagate_properties
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.mantid.mantid_algorithms import CloneWorkspace
from mslice.util.mantid import mantid_algorithms

from mslice.util.numpy_helper import apply_with_swapped_axes
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.workspace.workspace import Workspace

KB_MEV = constants.value('Boltzmann constant in eV/K') * 1000
E_TO_K = np.sqrt(2 * constants.neutron_mass * constants.elementary_charge / 1000) / constants.hbar


def compute_slice(selected_workspace, x_axis, y_axis, norm_to_one):
    workspace = get_workspace_handle(selected_workspace)
    slice = mantid_algorithms.Slice(OutputWorkspace='__' + workspace.name, InputWorkspace=workspace,
                                    XAxis=x_axis.to_dict(), YAxis=y_axis.to_dict(), PSD=workspace.is_PSD,
                                    EMode=workspace.e_mode, NormToOne=norm_to_one)
    propagate_properties(workspace, slice)
    if norm_to_one:
        slice = _norm_to_one(slice)
    return slice


def axis_values(axis):
    """Compute a numpy array of bins for the given axis values"""
    return np.linspace(axis.start_meV, axis.end_meV, get_number_of_steps(axis))


def compute_boltzmann_dist(sample_temp, delta_e):
    """calculates exp(-E/kBT), a common factor in intensity corrections"""
    kBT = sample_temp * KB_MEV
    return np.exp(-delta_e / kBT)


def compute_chi(scattering_data, sample_temp, e_axis):
    """
    :param scattering_data: Scattering data workspace
    :param sample_temp: The sample temperature in Kelvin
    :param e_axis: Axis object defining energy axis details
    :return: The dynamic susceptibility of the data
    """
    energy_transfer = axis_values(e_axis)
    signs = np.sign(energy_transfer)
    signs[signs == 0] = 1
    boltzmann_dist = compute_boltzmann_dist(sample_temp, energy_transfer)
    chi = np.pi * (signs + (boltzmann_dist * -signs))
    out = scattering_data * chi
    return out


def compute_chi_magnetic(chi):
    # 291 milibarns is the total neutron cross-section for a moment of one bohr magneton
    chi_magnetic = chi / 291
    return chi_magnetic


def compute_d2sigma(scattering_data, e_axis):
    """
    :param scattering_data: Scattering data workspace
    :param e_axis: Axis object defining energy axis details
    :return: d2sigma
    """
    Ei = scattering_data.e_fixed
    if Ei is None:
        return None
    ki = np.sqrt(Ei) * E_TO_K
    energy_transfer = axis_values(e_axis)
    kf = (np.sqrt(Ei - energy_transfer)*E_TO_K)
    return scattering_data * kf / ki


def compute_symmetrised(scattering_data, sample_temp, e_axis, data_rotated):
    energy_transfer = axis_values(e_axis)
    negative_de = energy_transfer[energy_transfer < 0]
    negative_de_len = len(negative_de)
    boltzmann_dist = compute_boltzmann_dist(sample_temp, negative_de)
    signal = scattering_data.get_signal()
    if data_rotated and scattering_data.is_PSD:
        new_signal = apply_with_swapped_axes(partial(modify_part_of_signal, boltzmann_dist, negative_de_len), signal)
    else:
        new_signal = modify_part_of_signal(boltzmann_dist, negative_de_len, signal)
    new_ws = CloneWorkspace(InputWorkspace=scattering_data, OutputWorkspace=scattering_data.name, store=False)
    propagate_properties(scattering_data, new_ws)
    new_ws.set_signal(new_signal)
    return new_ws


def modify_part_of_signal(multiplier, up_to_index, signal):
    lhs = signal[:, :up_to_index] * multiplier
    rhs = signal[:, up_to_index:]
    return np.concatenate((lhs, rhs), 1)


def compute_gdos(scattering_data, sample_temp, q_axis, e_axis):
    energy_transfer = axis_values(e_axis)
    momentum_transfer = axis_values(q_axis)
    momentum_transfer = np.square(momentum_transfer, out=momentum_transfer)
    boltzmann_dist = compute_boltzmann_dist(sample_temp, energy_transfer)
    gdos = scattering_data / momentum_transfer
    gdos *= energy_transfer
    gdos *= (1 - boltzmann_dist)
    return gdos


def sample_temperature(ws_name, sample_temp_fields):
    ws = get_workspace_handle(ws_name).raw_ws
    sample_temp = None
    for field_name in sample_temp_fields:
        try:
            sample_temp = ws.run().getLogData(field_name).value
        except RuntimeError:
            pass
        except AttributeError:
            sample_temp = ws.getExperimentInfo(0).run().getLogData(field_name).value
    try:
        float(sample_temp)
    except (ValueError, TypeError):
        pass
    else:
        return sample_temp
    if isinstance(sample_temp, string_types):
        sample_temp = get_sample_temperature_from_string(sample_temp)
    if isinstance(sample_temp, np.ndarray) or isinstance(sample_temp, list):
        sample_temp = np.mean(sample_temp)
    return sample_temp


def compute_recoil_line(ws_name, axis, relative_mass=None):
    efixed = get_workspace_handle(ws_name).e_fixed
    x_axis = np.arange(axis.start, axis.end, axis.step)
    if not relative_mass:
        relative_mass = 1
    if is_momentum(axis.units):
        momentum_transfer = x_axis
        line = np.square(momentum_transfer * 1.e10 * constants.hbar) / (2 * relative_mass * constants.neutron_mass) /\
            (constants.elementary_charge / 1000)
    elif is_twotheta(axis.units):
        tth = x_axis * np.pi / 180.
        if 'Direct' in get_workspace_handle(ws_name).e_mode:
            line = efixed * (2 - 2 * np.cos(tth)) / (relative_mass + 1 - np.cos(tth))
        else:
            line = efixed * (2 - 2 * np.cos(tth)) / (relative_mass - 1 + np.cos(tth))
    else:
        raise RuntimeError("units of axis not recognised")
    return x_axis, line


def _norm_to_one(workspace):
    return workspace / np.nanmax(np.abs(workspace.get_signal()))


def is_sliceable(workspace):
    ws = get_workspace_handle(workspace)
    if isinstance(ws, PixelWorkspace):
        return True
    else:
        validator = WorkspaceUnitValidator('DeltaE')
        try:
            isvalid = isinstance(ws, Workspace) and validator.isValid(ws.raw_ws) == ''
        except RuntimeError:
            return False
        return isvalid
