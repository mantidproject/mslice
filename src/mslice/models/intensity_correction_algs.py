import numpy as np

from scipy import constants

from mslice.models.alg_workspace_ops import get_number_of_steps
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.workspace.pixel_workspace import PixelWorkspace, HistogramWorkspace
from mslice.models.units import get_sample_temperature_from_string
from mslice.models.axis import Axis
from mslice.util.mantid.mantid_algorithms import CreateMDHistoWorkspace
from mslice.models.slice.slice_functions import compute_slice
from mslice.models.labels import is_momentum, is_twotheta
from mslice.models.cut.cut_algorithm import _cut_nonPSD_general
from math import trunc, ceil


KB_MEV = constants.value("Boltzmann constant in eV/K") * 1000
E_TO_K = (
    np.sqrt(2 * constants.neutron_mass * constants.elementary_charge / 1000)
    / constants.hbar
)
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
    kf = np.sqrt(e_fixed - energy_transfer) * E_TO_K
    return scattering_data * (kf / ki)


def compute_symmetrised(scattering_data, sample_temp, e_axis, data_rotated):
    energy_transfer = axis_values(e_axis)
    negative_de = energy_transfer[energy_transfer < 0]
    negative_de_len = len(negative_de)
    boltzmann_dist = compute_boltzmann_dist(sample_temp, negative_de)
    signal = scattering_data.get_signal()

    if data_rotated and scattering_data.is_PSD:
        transposed_signal = np.transpose(signal)
        signal_modification_array = generate_modification_array(
            boltzmann_dist, negative_de_len, transposed_signal
        )
        symm_workspace = scattering_data * np.transpose(signal_modification_array)
    else:
        signal_modification_array = generate_modification_array(
            boltzmann_dist, negative_de_len, signal
        )
        symm_workspace = scattering_data * signal_modification_array
    return symm_workspace


def generate_modification_array(multiplier, up_to_index, signal):
    modification_array = np.ones_like(signal)
    if len(signal.shape) < 2:  # cut
        lhs = modification_array[:up_to_index] * multiplier
        rhs = modification_array[up_to_index:]
        axis_index = 0
    else:  # slice
        lhs = modification_array[:, :up_to_index] * multiplier
        rhs = modification_array[:, up_to_index:]
        axis_index = 1
    return np.concatenate((lhs, rhs), axis_index)


def slice_compute_gdos(scattering_data, sample_temp, q_axis, e_axis, rotated):
    x_units = e_axis.units if rotated else q_axis.units
    x_units_scale = e_axis.scale if rotated else q_axis.scale
    if "DeltaE" in x_units and x_units_scale != 1.0:
        scattering_data = scattering_data*x_units_scale
    
    x_dim, y_dim = _get_slice_dimensions(scattering_data, x_units)
    x_dim_shape_index = (
        0 if x_dim.name == scattering_data._raw_ws.getXDimension().name else 1
    )
    y_dim_shape_index = (
        0 if y_dim.name == scattering_data._raw_ws.getXDimension().name else 1
    )
    signal = (
        scattering_data.get_signal()
        if isinstance(scattering_data, HistogramWorkspace)
        else scattering_data.get_signal().transpose()
    )
    n_bins_energy = (
        signal.shape[x_dim_shape_index] if rotated else signal.shape[y_dim_shape_index]
    )
    n_bins_momentum = (
        signal.shape[y_dim_shape_index] if rotated else signal.shape[x_dim_shape_index]
    )
    # The expression for the Boltzmann factor and DOS energy scaling refer to the phonon energy
    # which is the absolute value of the energy transfer (negative energy transfer is phonon anihilation).
    energy = np.abs(axis_values(e_axis, n_bins_energy))
    momentum_transfer = axis_values(q_axis, n_bins_momentum)
    momentum_transfer = np.square(momentum_transfer, out=momentum_transfer)
    boltzmann_dist = compute_boltzmann_dist(sample_temp, energy)
    gdos = scattering_data / momentum_transfer
    gdos *= energy
    gdos *= 1 - boltzmann_dist
    
    return gdos


def cut_compute_gdos(
    scattering_data,
    sample_temp,
    q_axis,
    e_axis,
    rotated,
    norm_to_one,
    algorithm,
    is_icut,
):
    original_data_ws = get_workspace_handle(scattering_data.parent)
    if isinstance(original_data_ws, PixelWorkspace):
        return _cut_compute_gdos_pixel(
            scattering_data,
            sample_temp,
            q_axis,
            e_axis,
            rotated,
            norm_to_one,
            algorithm,
            is_icut,
        )
    else:
        return _cut_compute_gdos(
            scattering_data,
            sample_temp,
            q_axis,
            e_axis,
            rotated,
            norm_to_one,
            algorithm,
            is_icut,
        )


def _cut_compute_gdos(
    scattering_data,
    sample_temp,
    q_axis,
    e_axis,
    rotated,
    norm_to_one,
    algorithm,
    is_icut,
):
    parent_ws = get_workspace_handle(scattering_data.parent)
    # Take a slice from parent_ws with bins to match cut, then gdos correct
    rebin_slice_q_axis, rebin_slice_e_axis = _get_rebin_slice_q_and_e_axis(
        parent_ws, q_axis, e_axis, is_icut
    )
    
    rebin_slice_gdos = _rebin_slice_and_gdos_correct(
        parent_ws,
        sample_temp,
        rebin_slice_q_axis,
        rebin_slice_e_axis,
        rotated,
        norm_to_one,
    )

    cut_axis = rebin_slice_e_axis if rotated else rebin_slice_q_axis
    int_axis = rebin_slice_q_axis if rotated else rebin_slice_e_axis
    cut_axis_id = 0 if rotated else 1
    return _reduce_bins_along_int_axis(
        rebin_slice_gdos,
        algorithm,
        cut_axis,
        int_axis,
        cut_axis_id,
        True,
        scattering_data.name,
    )


def _cut_compute_gdos_pixel(
    scattering_data,
    sample_temp,
    q_axis,
    e_axis,
    rotated,
    norm_to_one,
    algorithm,
    is_icut,
):
    pixel_ws = get_workspace_handle(scattering_data.parent)
    if is_icut:
        slice_ws = get_workspace_handle("__" + scattering_data.parent)
        slice_rotated = not _is_momentum_or_two_theta(
            slice_ws.raw_ws.getXDimension().getUnits()
        )  # fn arg rotated refers to cut.
    else:
        slice_rotated = not _is_momentum_or_two_theta(
            pixel_ws.raw_ws.getXDimension().getUnits()
        )  # no pre existing slice, use pixel ws.

    # Take a slice from initial pixel ws with bins to match cut, then gdos correct
    rebin_slice_q_axis, rebin_slice_e_axis = _get_rebin_slice_q_and_e_axis(
        pixel_ws, q_axis, e_axis, is_icut
    )
    rebin_slice_gdos = _rebin_slice_and_gdos_correct(
        pixel_ws,
        sample_temp,
        rebin_slice_q_axis,
        rebin_slice_e_axis,
        slice_rotated,
        norm_to_one,
    )

    cut_axis = rebin_slice_e_axis if rotated else rebin_slice_q_axis
    int_axis = rebin_slice_q_axis if rotated else rebin_slice_e_axis
    cut_slice_alignment = slice_rotated == rotated
    cut_axis_id = 1 if cut_slice_alignment else 0
    return _reduce_bins_along_int_axis(
        rebin_slice_gdos,
        algorithm,
        cut_axis,
        int_axis,
        cut_axis_id,
        cut_slice_alignment,
        scattering_data.name,
    )


def _get_rebin_slice_q_and_e_axis(workspace, q_axis, e_axis, is_icut):
    q_limits = workspace.limits[q_axis.units]
    e_limits = workspace.limits[e_axis.units]
    rebin_slice_q_axis = _get_slice_axis(q_limits, q_axis, is_icut)
    rebin_slice_e_axis = _get_slice_axis(e_limits, e_axis, is_icut)
    return rebin_slice_q_axis, rebin_slice_e_axis


def _rebin_slice_and_gdos_correct(
    workspace, sample_temp, rebin_slice_q_axis, rebin_slice_e_axis, rotated, norm_to_one
):
    rebin_slice_x_axis = rebin_slice_e_axis if rotated else rebin_slice_q_axis
    rebin_slice_y_axis = rebin_slice_q_axis if rotated else rebin_slice_e_axis
    rebin_slice = compute_slice(
        workspace,
        rebin_slice_x_axis,
        rebin_slice_y_axis,
        norm_to_one,
        store_in_ADS=False,
    )

    rebin_slice_gdos = slice_compute_gdos(
        rebin_slice, sample_temp, rebin_slice_q_axis, rebin_slice_e_axis, rotated
    )
    return rebin_slice_gdos


def _get_slice_axis(pixel_limits, cut_axis, is_icut):
    slice_step_size = pixel_limits[2]
    if is_icut:  # avoid loss of resolution by aligning icut with slice bins
        data_start = pixel_limits[0]
        steps_before_cut = trunc((cut_axis.start - data_start) / slice_step_size)
        step_aligned_cut_start = round(
            data_start + steps_before_cut * slice_step_size, 8
        )

        steps_to_cut_end = ceil((cut_axis.end - data_start) / slice_step_size)
        step_aligned_cut_end = round(data_start + steps_to_cut_end * slice_step_size, 8)
        ret_axis = Axis(
            cut_axis.units,
            step_aligned_cut_start,
            step_aligned_cut_end,
            slice_step_size,
            cut_axis.e_unit,
        )
    else:  # if not icut (user specified), retain user input unless smaller than data steps.
        step_size = cut_axis.step if cut_axis.step != 0 else pixel_limits[2]
        if step_size < slice_step_size:
            step_size = slice_step_size
        ret_axis = Axis(
            cut_axis.units, cut_axis.start, cut_axis.end, step_size, cut_axis.e_unit
        )
    return ret_axis


def _reduce_bins_along_int_axis(
    slice_gdos,
    algorithm,
    cut_axis,
    int_axis,
    cut_axis_id,
    cut_slice_alignment,
    output_name,
):
    if isinstance(slice_gdos, HistogramWorkspace):
        signal, error = _reduce_bins_and_return_signal_error_PSD(
            slice_gdos, algorithm, int_axis, cut_axis_id, cut_slice_alignment
        )
    else:
        signal, error = _reduce_bins_and_return_signal_error(
            slice_gdos, algorithm, cut_axis, int_axis, cut_axis_id
        )

    int_axis_id = 0 if cut_axis_id else 1
    slice_x, slice_y = _get_slice_dimensions(slice_gdos, cut_axis.units)
    x_dim = slice_x if cut_slice_alignment else slice_y
    y_dim = slice_y if cut_slice_alignment else slice_x
    extents = (
        f"{y_dim.getMinimum()},{y_dim.getMaximum()},"
        f"{x_dim.getMinimum()},{x_dim.getMaximum()}"
    )
    no_of_bins = f"{signal.shape[cut_axis_id]},{signal.shape[int_axis_id]}"
    names = f"{x_dim.name},{y_dim.name}"
    units = f"{x_dim.getUnits()},{y_dim.getUnits()}"

    new_ws = CreateMDHistoWorkspace(
        OutputWorkspace=output_name,
        Dimensionality=2,
        Extents=extents,
        SignalInput=signal,
        ErrorInput=error,
        NumberOfBins=no_of_bins,
        Names=names,
        Units=units,
        StoreInADS=False,
    )

    int_axis.step = 0
    new_ws.axes = (cut_axis, int_axis)
    return new_ws


def _reduce_bins_and_return_signal_error(
    slice_gdos, algorithm, cut_axis, int_axis, cut_axis_id
):
    ax1 = f"{cut_axis.start}, {cut_axis.step}, {cut_axis.end}"
    ax2 = f"{int_axis.start}, {int_axis.end - int_axis.start}, {int_axis.end}"

    if cut_axis_id == 0:
        ws_out = _cut_nonPSD_general(ax1, ax2, slice_gdos.raw_ws, algorithm)
    else:
        ws_out = _cut_nonPSD_general(ax2, ax1, slice_gdos.raw_ws, algorithm)
    signal = ws_out.extractY()
    if "DeltaE" in cut_axis.units and cut_axis.scale != 1.0:
        error = np.sqrt(np.nansum(slice_gdos.get_variance(), cut_axis_id, keepdims=True))
    else:
        error = ws_out.extractE()
        
    return signal, error


def _reduce_bins_and_return_signal_error_PSD(
    slice_gdos, algorithm, int_axis, cut_axis_id, cut_slice_alignment
):
    # PSD type workspace, just sum all the bins
    signal = np.nansum(slice_gdos.get_signal(), axis=cut_axis_id, keepdims=True)
    error = np.sqrt(np.nansum(slice_gdos.get_variance(), cut_axis_id, keepdims=True))
    if not cut_slice_alignment:
        signal = signal.transpose()
        error = error.transpose()
    if algorithm == "Integration":
        signal = signal * int_axis.step
        error = error * int_axis.step
    return signal, error


def _get_slice_dimensions(slice, x_units):
    x_is_momentum = _is_momentum_or_two_theta(x_units)
    dim1 = slice._raw_ws.getXDimension()
    dim2 = slice._raw_ws.getYDimension()
    if (x_is_momentum and _is_momentum_or_two_theta(dim1.getUnits())) or (
        not x_is_momentum and not _is_momentum_or_two_theta(dim1.getUnits())
    ):
        ret_val = (dim1, dim2)
    else:
        ret_val = (dim2, dim1)
    return ret_val


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
    if isinstance(sample_temp, str):
        sample_temp = get_sample_temperature_from_string(sample_temp)
    elif isinstance(sample_temp, np.ndarray) or isinstance(sample_temp, list):
        sample_temp = np.mean(sample_temp)
    else:
        float(sample_temp)
    return sample_temp


def _is_momentum_or_two_theta(units):
    if is_momentum(units) or is_twotheta(units):
        return True
    else:
        return False
