from __future__ import (absolute_import, division, print_function)
from six import string_types
import numpy as np

from scipy import constants

from mslice.models.alg_workspace_ops import get_number_of_steps
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.models.units import get_sample_temperature_from_string
from mslice.models.axis import Axis
from mslice.util.mantid.mantid_algorithms import CloneWorkspace, CreateMDHistoWorkspace
from mslice.util.numpy_helper import apply_with_swapped_axes, transform_array_to_workspace
from mslice.models.cut.cut_functions import compute_cut
from mslice.models.slice.slice_functions import compute_slice
from mslice.models.labels import is_momentum
from math import trunc, ceil


KB_MEV = constants.value('Boltzmann constant in eV/K') * 1000
E_TO_K = np.sqrt(2 * constants.neutron_mass * constants.elementary_charge / 1000) / constants.hbar
CHI_MAGNETIC_CONST = 291  # 291 milibarns is the total neutron cross-section for a moment of one bohr magneton


def compute_boltzmann_dist(sample_temp, delta_e):
    """calculates exp(-E/kBT), a common factor in intensity corrections"""
    kBT = sample_temp * KB_MEV
    return np.exp(-delta_e / kBT)


def axis_values(axis, step_num=False):
    """Compute a numpy array of bins for the given axis values"""
    step_num = step_num if step_num else get_number_of_steps(axis)
    return np.linspace(axis.start_meV, axis.end_meV, step_num)


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
        transposed_signal = np.transpose(signal)
        signal_modification_array = generate_modification_array(boltzmann_dist, negative_de_len, transposed_signal)
        symm_workspace = scattering_data * np.transpose(signal_modification_array)
    else:
        signal_modification_array = generate_modification_array(boltzmann_dist, negative_de_len, signal)
        symm_workspace = scattering_data * signal_modification_array
    return symm_workspace


def generate_modification_array(multiplier, up_to_index, signal):
    modification_array = np.ones_like(signal)
    if len(signal.shape) < 2:  #cut
        lhs = modification_array[:up_to_index] * multiplier
        rhs = modification_array[up_to_index:]
        axis_index = 0
    else:  #slice
        lhs = modification_array[:, :up_to_index] * multiplier
        rhs = modification_array[:, up_to_index:]
        axis_index = 1
    return np.concatenate((lhs, rhs), axis_index)


def slice_compute_gdos(scattering_data, sample_temp, q_axis, e_axis, rotated):
    n_bins_energy = scattering_data.get_signal().shape[0] if rotated else scattering_data.get_signal().shape[1]
    n_bins_momentum = scattering_data.get_signal().shape[1] if rotated else scattering_data.get_signal().shape[0]
    energy_transfer = axis_values(e_axis, n_bins_energy)
    momentum_transfer = axis_values(q_axis, n_bins_momentum)
    momentum_transfer = np.square(momentum_transfer, out=momentum_transfer)
    boltzmann_dist = compute_boltzmann_dist(sample_temp, energy_transfer)
    gdos = scattering_data / momentum_transfer
    gdos *= energy_transfer
    gdos *= (1 - boltzmann_dist)
    return gdos


def cut_compute_gdos(scattering_data, sample_temp, q_axis, e_axis, rotated, norm_to_one, algorithm):
    original_data_ws = get_workspace_handle(scattering_data.parent)
    if isinstance(original_data_ws, PixelWorkspace):
        return _cut_compute_gdos_pixel(original_data_ws, sample_temp, q_axis, e_axis, rotated, norm_to_one, algorithm)
    else:
        parent_slice = get_workspace_handle("__" + scattering_data.parent)
        return _cut_compute_gdos(parent_slice, sample_temp, q_axis, e_axis, rotated, norm_to_one, algorithm)


def _cut_compute_gdos(parent_ws, sample_temp, q_axis, e_axis, rotated, norm_to_one, algorithm):
    q_limits = parent_ws.limits[q_axis.units]
    e_limits = parent_ws.limits[e_axis.units]
    slice_q_axis = Axis(q_axis.units, q_limits[0], q_limits[1], q_limits[2], q_axis.e_unit)
    slice_e_axis = Axis(e_axis.units, e_limits[0], e_limits[1], e_limits[2], e_axis.e_unit)
    slice_gdos = slice_compute_gdos(parent_ws, sample_temp, slice_q_axis, slice_e_axis, rotated)
    cut_axis = e_axis if rotated else q_axis
    int_axis = q_axis if rotated else e_axis
    return compute_cut(slice_gdos, cut_axis, int_axis, norm_to_one, algorithm)


def _cut_compute_gdos_pixel(parent_ws, sample_temp, q_axis, e_axis, rotated, norm_to_one, algorithm):
    q_limits = parent_ws.limits[q_axis.units]
    e_limits = parent_ws.limits[e_axis.units]

    slice_q_axis = _get_slice_axis(q_limits, q_axis)
    slice_e_axis = _get_slice_axis(e_limits, e_axis)

    x_is_momentum = is_momentum(parent_ws.raw_ws.getXDimension().getUnits())
    slice_x_axis = slice_q_axis if x_is_momentum else slice_e_axis
    slice_y_axis = slice_e_axis if x_is_momentum else slice_q_axis
    rebin_slice = compute_slice(parent_ws, slice_x_axis, slice_y_axis, norm_to_one)

    slice_gdos = slice_compute_gdos(rebin_slice, sample_temp, slice_q_axis, slice_e_axis, rotated=False) #rotation already accounted for

    cut_axis = slice_e_axis if rotated else slice_q_axis
    int_axis = slice_q_axis if rotated else slice_e_axis
    return _reduce_bins_along_int_axis(slice_gdos, algorithm, cut_axis, int_axis, rotated)


def _get_slice_axis(slice_limits, cut_axis):
    data_start = slice_limits[0]
    step_size = slice_limits[2]

    steps_before_cut = trunc((cut_axis.start - data_start) / step_size)
    step_aligned_cut_start = data_start + steps_before_cut * step_size

    steps_to_cut_end = ceil((cut_axis.end - data_start) / step_size)
    step_aligned_cut_end = data_start + steps_to_cut_end * step_size

    return Axis(cut_axis.units, step_aligned_cut_start, step_aligned_cut_end, step_size, cut_axis.e_unit)


def _reduce_bins_along_int_axis(slice_gdos, algorithm, cut_axis, int_axis, rotated):
    axis_id = 0 if rotated else 1
    signal_array_adj = _adjust_first_and_last_bins(slice_gdos._raw_ws.getSignalArray())
    signal = signal_array_adj.sum(axis=axis_id, keepdims=True)
    error_array_adj = _adjust_first_and_last_bins(slice_gdos._raw_ws.getErrorSquaredArray())
    error_squared = error_array_adj.sum(axis=axis_id, keepdims=True)
    if not rotated:
        signal = signal.transpose()
        error_squared = error_squared.transpose()
    if algorithm == 'Integration':
        signal = signal * int_axis.step

    x_dim = slice_gdos._raw_ws.getXDimension() if not rotated else slice_gdos._raw_ws.getYDimension()
    y_dim = slice_gdos._raw_ws.getYDimension() if not rotated else slice_gdos._raw_ws.getXDimension()
    extents = f"{y_dim.getMinimum()},{y_dim.getMaximum()}," \
              f"{x_dim.getMinimum()},{x_dim.getMaximum()}"
    no_of_bins = f"{signal.shape[0]},{signal.shape[1]}"
    names = f"{y_dim.name},{x_dim.name}"
    units = f"{y_dim.getUnits()},{x_dim.getUnits()}"

    new_ws = CreateMDHistoWorkspace(Dimensionality=2, Extents=extents, SignalInput=signal, ErrorInput=error_squared,
                    NumberOfBins=no_of_bins, Names=names, Units=units)

    int_axis.step = 0
    new_ws.axes = (cut_axis, int_axis)
    return new_ws


def _adjust_first_and_last_bins(array):
    return array


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
