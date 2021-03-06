from mslice.workspace.histogram_workspace import HistogramWorkspace
from mslice.workspace.base import WorkspaceBase as Workspace
from mslice.workspace.workspace import Workspace as MatrixWorkspace
from mslice.models.alg_workspace_ops import get_axis_range, get_available_axes
from mslice.models.axis import Axis
from mslice.models.workspacemanager.workspace_provider import workspace_exists
from mslice.plotting.globalfiguremanager import GlobalFigureManager

_overplot_keys = {'Hydrogen': 1, 'Deuterium': 2, 'Helium': 4, 'Aluminium': 'Aluminium', 'Copper': 'Copper',
                  'Niobium': 'Niobium', 'Tantalum': 'Tantalum', 'Arbitrary Nuclei': 'Arbitrary Nuclei',
                  'CIF file': 'CIF file'}

_function_to_intensity = {
    'show_scattering_function': 's(q,e)',
    'show_dynamical_susceptibility': 'chi',
    'show_dynamical_susceptibility_magnetic': 'chi_mag',
    'show_d2sigma': 'xsec',
    'show_symmetrised': 'symm',
    'show_gdos': 'gdos',
}

_intensity_to_action = {
    's(q,e)': 'action_sqe',
    'chi': 'action_chi_qe',
    'chi_mag': 'action_chi_qe_magnetic',
    'xsec': 'action_d2sig_dw_de',
    'symm': 'action_symmetrised_sqe',
    'gdos': 'action_gdos',
}

_intensity_to_workspace = {
    'chi': 'chi',
    'chi_mag': 'chi_magnetic',
    'xsec': 'd2sigma',
    'symm': 'symmetrised',
    'gdos': 'gdos',
}


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


def _rescale_energy_cut_plot(presenter, cuts, new_e_unit):
    """Given a CutPlotterPresenter and a set of cached cuts,
    rescales the workspaces to a different energy-unit and replot"""
    import copy
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
    Checks if args[0] is a HistogramWorkspace
    """
    if isinstance(args[0], HistogramWorkspace) and \
            sum([args[0].raw_ws.getDimension(i).getNBins() != 1 for i in range(args[0]._raw_ws.getNumDims())]) == 1:
        return True
    else:
        return False
