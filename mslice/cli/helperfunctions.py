import copy

from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.workspace.base import WorkspaceBase as Workspace
from mslice.workspace.workspace import Workspace as MatrixWorkspace
from mslice.models.alg_workspace_ops import get_axis_range, get_available_axes
from mslice.models.axis import Axis
from mslice.models.workspacemanager.workspace_provider import workspace_exists
from mslice.models.intensity_correction_algs import (compute_chi, compute_d2sigma,
                                                     compute_symmetrised, cut_compute_gdos)
from mslice.models.cut.cut import SampleTempValueError
from mslice.plotting.globalfiguremanager import GlobalFigureManager
from mslice.util.intensity_correction import IntensityType, IntensityCache

_overplot_keys = {'Hydrogen': 1, 'Deuterium': 2, 'Helium': 4, 'Aluminium': 'Aluminium', 'Copper': 'Copper',
                  'Niobium': 'Niobium', 'Tantalum': 'Tantalum', 'Arbitrary Nuclei': 'Arbitrary Nuclei',
                  'CIF file': 'CIF file'}


def _update_legend():
    plot_handler = GlobalFigureManager.get_active_figure().plot_handler
    plot_handler.update_legend()


def _update_overplot_checklist(key):
    if isinstance(key, int) and key > 4:
        plot_handler = GlobalFigureManager.get_active_figure().plot_handler
        plot_handler._arb_nuclei_rmm = key
        key = 'Arbitrary Nuclei'
    else:
        for element, value in _overplot_keys.items():
            if value == key:
                key = element

    window = GlobalFigureManager.get_active_figure().window
    getattr(window, 'action_' + key.replace(' ', '_').lower()).setChecked(True)


def _get_overplot_key(element, rmm):
    if element is not None and rmm is not None:
        raise RuntimeError('Cannot use both element name and relative molecular mass')
    if element is None and rmm is None:
        raise RuntimeError('An element name or relative molecular mass is required')

    if rmm is None:
        return _overplot_keys[element.capitalize()]
    return rmm


def _string_to_axis(string):
    axis = string.split(',')
    if len(axis) != 4 and len(axis) != 5:
        raise ValueError('axis should be specified in format <name>,<start>,<end>,<step_size>(,<e_unit>)')
    return Axis(axis[0], axis[1], axis[2], axis[3]) if len(axis) == 4 else Axis(*axis)


def _string_to_integration_axis(string):
    """Allows step to be omitted and set to default value"""
    axis_str = string.split(',')
    if len(axis_str) < 3:
        raise ValueError('axis should be specified in format <name>,<start>,<end>(,<step>,<e_unit>)')
    elif len(axis_str) == 3:
        valid_axis = Axis(axis_str[0], axis_str[1], axis_str[2], str(float(axis_str[2]) - float(axis_str[1])))
    elif len(axis_str) == 4:
        valid_axis = Axis(axis_str[0], axis_str[1], axis_str[2], axis_str[3])
    else:
        valid_axis = Axis(axis_str[0], axis_str[1], axis_str[2], axis_str[3], axis_str[4])
    return valid_axis


def _process_axis(axis, fallback_index, input_workspace, string_function=_string_to_axis):
    available_axes = get_available_axes(input_workspace)
    if axis is None:
        axis = available_axes[fallback_index]
    # check to see if axis is just a name e.g 'DeltaE' or a full binning spec e.g. 'DeltaE,0,1,100'
    if ',' in axis:
        axis = string_function(axis)
    elif axis in available_axes:
        range = get_axis_range(input_workspace, axis)
        range = list(map(float, range))
        axis = Axis(units=axis, start=range[0], end=range[1], step=range[2])
    else:
        raise RuntimeError("Axis '%s' not recognised. Workspace has these axes: %s " %
                           (axis, ', '.join(available_axes)))
    return axis


def _check_workspace_name(workspace):
    if isinstance(workspace, Workspace):
        return
    if not isinstance(workspace, str):
        raise TypeError('InputWorkspace must be a workspace or a workspace name')
    if not workspace_exists(workspace):
        raise TypeError('InputWorkspace %s could not be found.' % workspace)


def _check_workspace_type(workspace, correct_type):
    """Check a PSD workspace is MatrixWorkspace, or non-PSD is the specified type"""
    if workspace.is_PSD:
        if isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type - run MakeProjection first.")
        if not isinstance(workspace, correct_type):
            raise RuntimeError("Incorrect workspace type.")
    else:
        if not isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type.")

def _get_workspace_type(workspace):
    """Determine workspace type"""
    if isinstance(workspace, MatrixWorkspace):
        return "MatrixWorkspace"
    if isinstance(workspace, HistogramWorkspace):
        return "HistogramWorkspace"

def _rescale_energy_cut_plot(presenter, cuts, new_e_unit):
    """Given a CutPlotterPresenter and a set of cached cuts,
    rescales the workspaces to a different energy-unit and replot"""
    cuts_copy = copy.deepcopy(cuts)  # Because run_cut will overwrite the cuts cache for plot_over=True
    for id, cut in enumerate(cuts_copy):
        cut.cut_axis.e_unit = new_e_unit
        presenter.run_cut(cut.workspace_raw_name, cut, plot_over=(id > 0))


# Arguments Validation
def is_slice(*args):
    """
    Checks if args[0] is a WorkspaceBase or HistogramWorkspace
    """
    if not isinstance(args[0], Workspace) or is_cut(*args):
        return False
    if isinstance(args[0], Workspace) or args[0]._raw_ws.getNumDims() == 2:
        return True


def is_cut(*args):
    """
    Checks if args[0] is a HistogramWorkspace and if the bin number matches
    """
    if isinstance(args[0], HistogramWorkspace) and \
            sum([args[0].raw_ws.getDimension(i).getNBins() != 1 for i in range(args[0]._raw_ws.getNumDims())]) == 1:
        return True
    else:
        return False

def is_hs_workspace(*args):
    """
    Checks if args[0] is a HistogramWorkspace
    """
    return isinstance(args[0], HistogramWorkspace)

def append_visible_handle(visible_handles, handles, idx: int):
    handle = handles[idx]
    visible_handles.append(handle[0])
    return None


def append_visible_label(visible_labels, labels, idx: int):
    label = labels[idx]
    visible_labels.append(label)
    return None


def append_visible_handle_and_label(visible_handles, handles, visible_labels, labels, idx: int):
    append_visible_handle(visible_handles, handles, idx)
    append_visible_label(visible_labels, labels, idx)
    return None


def show_or_hide_errorbars_of_a_line(container, alpha: float):
    elements = container.get_children()[1:]
    for element in elements:
        element.set_alpha(alpha)
    return None


def show_or_hide_a_line(container, show_or_hide: bool):
    line = container.get_children()[0]
    line.set_visible(show_or_hide)
    return None


def hide_a_line_and_errorbars(ax, idx: int):
    container = ax.containers[idx]
    show_or_hide_a_line(container, False)
    show_or_hide_errorbars_of_a_line(container, 0.0)
    return None


def _correct_intensity(scattering_data, intensity_correction, e_axis, q_axis, NormToOne, Algorithm, rotated, sample_temp):
    try:
        intensity_correction = IntensityCache.get_intensity_type_from_desc(intensity_correction)
    except ValueError:
        raise ValueError(f"Input intensity correction invalid: {intensity_correction}")

    if intensity_correction == IntensityType.SCATTERING_FUNCTION:
        return scattering_data
    elif intensity_correction == IntensityType.CHI:
        _check_sample_temperature(sample_temp, scattering_data.name)
        return compute_chi(scattering_data, sample_temp, e_axis)
    elif intensity_correction == IntensityType.CHI_MAGNETIC:
        _check_sample_temperature(sample_temp, scattering_data.name)
        return compute_chi(scattering_data, sample_temp, e_axis, True)
    elif intensity_correction == IntensityType.D2_SIGMA:
        return compute_d2sigma(scattering_data, e_axis, scattering_data.e_fixed)
    elif intensity_correction == IntensityType.SYMMETRISED:
        _check_sample_temperature(sample_temp, scattering_data.name)
        return compute_symmetrised(scattering_data, sample_temp, e_axis, rotated)
    elif intensity_correction == "gdos":
        _check_sample_temperature(sample_temp, scattering_data.name)
        return cut_compute_gdos(scattering_data, sample_temp, q_axis, e_axis, rotated, NormToOne, Algorithm)


def _check_sample_temperature(sample_temperature, workspace_name):
    throw_error = False
    if sample_temperature is not None:
        try:
            sample_temperature = float(sample_temperature)
        except ValueError:
            throw_error = True
        if sample_temperature < 0:
            throw_error = True
    if throw_error:
        raise SampleTempValueError('No, or invalid, sample temperature provided', workspace_name)
