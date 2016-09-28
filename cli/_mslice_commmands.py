# Helper tools
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider as _MantidWorkspaceProvider
from presenters.slice_plotter_presenter import Axis as _Axis
from mantid.kernel.funcinspect import lhs_info as _lhs_info

_workspace_provider = _MantidWorkspaceProvider()

def _process_axis(axis, fallback_index, input_workspace):
    if axis is None:
        axis = _slice_algorithm.get_available_axis(input_workspace)[fallback_index]

    # check to see if axis is just a name e.g 'DeltaE' or a full binning spec e.g. 'DeltaE,0,1,100'
    if ',' in axis:
        axis = _string_to_axis(axis)
    else:
        axis = _Axis(units=axis, start=None, end=None, step=None) # The model will fill in the rest


    return axis

def _string_to_axis(string):
    axis = string.split(',')
    if len(axis) != 4:
        raise ValueError('axis should be specified in format <name>,<start>,<end>,<step_size>')
    name = axis[0].strip()
    try:
        start = float(axis[1])
    except:
        raise ValueError("start '%s' is not a valid float"%axis[1])

    try:
        end = float(axis[2])
    except:
        raise ValueError("end '%s' is not a valid float"%axis[2])

    try:
        step = float(axis[3])
    except:
        raise ValueError("step '%s' is not a valid float"%axis[3])
    return _Axis(name, start, end, step)

# Mantid Tools
from mantid.simpleapi import mtd, Load, ConvertUnits, RenameWorkspace

# Projections
from models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator as _MantidProjectionCalculator
from mantid.api import Workspace as _Workspace
_powder_projection_model = _MantidProjectionCalculator()


def get_projection(input_workspace, axis1, axis2):
    """ Calculate projections of workspace.

    Keyword Arguments:
        input_workspace -- Workspace to project, can be either python handle to workspace or a string containing the
        workspace name.
        axis1 -- The first axis of projection (string)
        axis2 -- The second axis of the projection (string)

    """
    if isinstance(input_workspace, _Workspace):
        input_workspace = input_workspace.getName()
    output_workspace = _powder_projection_model.calculate_projection(input_workspace=input_workspace, axis1=axis1,
                                                                     axis2=axis2)
    try:
        names = _lhs_info('names')
    except:
        names = [output_workspace.getName()]
    if len(names) > 1:
        raise Exception('Too many left hand side arguments, %s' % str(names))
    RenameWorkspace(InputWorkspace=output_workspace, OutputWorkspace=names[0])
    return output_workspace

#Slicing
from models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter as _MatplotlibSlicePlotter
from models.slice.mantid_slice_algorithm import MantidSliceAlgorithm as _MantidSliceAlgorithm
from mantid.api import IMDWorkspace as _IMDWorkspace

_slice_algorithm = _MantidSliceAlgorithm()
_slice_model = _MatplotlibSlicePlotter(_slice_algorithm)


def get_slice(input_workspace, x=None, y=None, ret_val='both', normalize=False):
    """ Get Slice from workspace as numpy array.

    Keyword Arguments:
    input_workspace -- The workspace to slice. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)

    x -- The x axis of the slice. If not specified will default to Dimension 0 of the workspace
    y -- The y axis of the slice. If not specified will default to Dimension 1 of the workspace
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.

    ret_val -- a string to specify the return value, if ret_val == 'slice' the function will return a single 2D numpy
    array containing the slice data. if ret_value == 'extents' it will return a list containing the range of the slice
    taken [xmin, xmax, ymin, ymax]. if ret_val == 'both' then it will return a tuple (<slice>, <extents>)

    normalize -- if set to True the slice will be normalize to one.

    """

    input_workspace = _workspace_provider.get_workspace_handle(input_workspace)
    assert isinstance(input_workspace, _IMDWorkspace)
    x_axis = _process_axis(x, 0, input_workspace)
    y_axis = _process_axis(y, 1, input_workspace)

    slice_array, extents = _slice_algorithm.compute_slice(selected_workspace=input_workspace, x_axis=x_axis,
                                                           smoothing=None, y_axis=y_axis, norm_to_one=normalize)
    if ret_val == 'slice':
        return slice_array
    elif ret_val == 'extents':
        return extents
    elif ret_val == 'both':
        return slice_array, extents
    else:
        raise ValueError("ret_val should be 'slice', 'extents' or 'both' and not '%s' " % ret_val)


def plot_slice(input_workspace, x=None, y=None, colormap='viridis', intensity_min=None, intensity_max=None,
               normalize=False):
    """ Plot slice from workspace

    Keyword Arguments:
    input_workspace -- The workspace to slice. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)

    x -- The x axis of the slice. If not specified will default to Dimension 0 of the workspace
    y -- The y axis of the slice. If not specified will default to Dimension 1 of the workspace
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.

    colormap -- a matplotlib colormap.
    intensity_min -- minimum value for intensity
    intensity_max -- maximum value for intensity

    normalize -- if set to True the slice will be normalize to one.

    """

    input_workspace = _workspace_provider.get_workspace_handle(input_workspace)
    assert isinstance(input_workspace, _IMDWorkspace)

    x_axis = _process_axis(x, 0, input_workspace)
    y_axis = _process_axis(y, 1, input_workspace)

    _slice_model.plot_slice(selected_workspace=input_workspace,x_axis=x_axis, y_axis=y_axis, colourmap=colormap,
                            intensity_start=intensity_min,intensity_end=intensity_max,
                            smoothing=None, norm_to_one=normalize)



# Cutting
from models.cut.mantid_cut_algorithm import MantidCutAlgorithm as _MantidCutAlgorithm
from models.cut.matplotlib_cut_plotter import MatplotlibCutPlotter
_cut_algorithm = _MantidCutAlgorithm()
_cut_plotter = MatplotlibCutPlotter(_cut_algorithm)


def get_cut_xye(input_workspace, cut_axis, integration_start, integration_end, normalize=False):
    """Return x, y and e of a cut of a workspace. The function will return a tuple of 3 single dimensional numpy arrays.

    Keyword Arguments
    input_workspace -- The workspace to cut. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)
    cut_axis -- The axis to cut along.
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.
    integration_start -- value to start integrating from
    integration_end -- The value to end the integration at
    normalize -- will normalize the cut data to one if set to true
    """
    if isinstance(input_workspace, _Workspace):
        input_workspace = input_workspace.getName()
    cut_axis = _process_axis(cut_axis, None, input_workspace)
    x, y, e = _cut_algorithm.compute_cut_xye(input_workspace, cut_axis, integration_start, integration_end, 
                                             is_norm=normalize)
    x, y, e = x.squeeze(), y.squeeze(), e.squeeze()
    return x, y, e


def plot_cut(input_workspace, cut_axis, integration_start, integration_end, intensity_start=None,
             intensity_end=None, normalize=False, hold=False):
    """Take a cut of the workspace and plot it.

    Keyword Arguments
    input_workspace -- The workspace to cut. Must be an MDWorkspace with 2 Dimensions. The parameter can be either a
    python handle to the workspace to slice OR the workspaces name in the ADS (string)
    cut_axis -- The axis to cut along.
    Axis Format:-
        Either a string in format '<name>, <start>, <end>, <step_size>' e.g. 'DeltaE,0,100,5'
        or just the name e.g. 'DeltaE'. That case the start and en will default to the range in the data.
    integration_start -- value to start integrating from
    integration_end -- The value to end the integration at
    normalize -- will normalize the cut data to one if set to true
    """
    if isinstance(input_workspace, _Workspace):
        input_workspace = input_workspace.getName()
    cut_axis = _process_axis(cut_axis, None, input_workspace)
    _cut_plotter.plot_cut(input_workspace, cut_axis, integration_start, integration_end, normalize, intensity_start,
                          intensity_end, plot_over=hold)