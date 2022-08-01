from __future__ import (absolute_import, division, print_function)
from six import string_types
import numpy as np
from functools import partial

from scipy import constants

from mslice.models.alg_workspace_ops import get_number_of_steps
from mslice.models.workspacemanager.workspace_algorithms import propagate_properties
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.models.units import get_sample_temperature_from_string
from mslice.util.mantid.mantid_algorithms import CloneWorkspace
from mslice.util.numpy_helper import apply_with_swapped_axes, transform_array_to_workspace


KB_MEV = constants.value('Boltzmann constant in eV/K') * 1000
E_TO_K = np.sqrt(2 * constants.neutron_mass * constants.elementary_charge / 1000) / constants.hbar
CHI_MAGNETIC_CONST = 291  # 291 milibarns is the total neutron cross-section for a moment of one bohr magneton


def compute_boltzmann_dist(sample_temp, delta_e):
    """calculates exp(-E/kBT), a common factor in intensity corrections"""
    kBT = sample_temp * KB_MEV
    return np.exp(-delta_e / kBT)


def axis_values(axis):
    """Compute a numpy array of bins for the given axis values"""
    return np.linspace(axis.start_meV, axis.end_meV, get_number_of_steps(axis))


def compute_chi(scattering_data, sample_temp, e_axis, magnetic=False):
    """
    :param scattering_data: Scattering data workspace
    :param sample_temp: The sample temperature in Kelvin
    :param e_axis: Axis object defining energy axis details
    :param magnetic: Flag to account for magnetic susceptibility
    :return: The dynamic (and optionally, magnetic) susceptibility of the data
    """
    energy_transfer = axis_values(e_axis)
    signs = np.sign(energy_transfer)
    signs[signs == 0] = 1
    boltzmann_dist = compute_boltzmann_dist(sample_temp, energy_transfer)
    chi = np.pi * (signs + (boltzmann_dist * -signs))
    mag_scale = CHI_MAGNETIC_CONST if magnetic else 1
    out = scattering_data * (chi / mag_scale)
    return out


def compute_d2sigma(scattering_data, e_axis, e_fixed):
    """
    :param scattering_data: Scattering data workspace
    :param e_axis: Axis object defining energy axis details
    :return: d2sigma
    """

    if e_fixed is None:
        return None
    ki = np.sqrt(e_fixed) * E_TO_K
    energy_transfer = axis_values(e_axis)
    kf = (np.sqrt(e_fixed - energy_transfer)*E_TO_K)
    return scattering_data * (kf / ki)


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
    new_signal = transform_array_to_workspace(new_signal, new_ws)
    new_ws.set_signal(new_signal)
    return new_ws


def modify_part_of_signal(multiplier, up_to_index, signal):
    if len(signal.shape) < 2:  #cut
        lhs = signal[:up_to_index] * multiplier
        rhs = signal[up_to_index:]
        axis_index = 0
    else:  #slice
        lhs = signal[:, :up_to_index] * multiplier
        rhs = signal[:, up_to_index:]
        axis_index = 1
    return np.concatenate((lhs, rhs), axis_index)


def slice_compute_gdos(scattering_data, sample_temp, q_axis, e_axis):
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
