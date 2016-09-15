# Helper tools
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from presenters.slice_plotter_presenter import Axis
from mantid.kernel.funcinspect import lhs_info

_workspace_provider = MantidWorkspaceProvider()

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
    return Axis(name, start, end, step)

# Mantid Tools
from mantid.simpleapi import mtd, Load, ConvertUnits, RenameWorkspace

# Projections
from models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator
from mantid.api import Workspace
_powder_projection_model = MantidProjectionCalculator()


def get_projection(input_workspace, axis1, axis2):
    if isinstance(input_workspace, Workspace):
        input_workspace = input_workspace.getName()
    output_workspace = _powder_projection_model.calculate_projection(input_workspace=input_workspace, axis1=axis1,
                                                                     axis2=axis2)
    try:
        names = lhs_info('names')
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


def get_slice(input_workspace, x=None, y=None, ret_val='both', normalized=False):
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

    normalized -- if set to True the slice will be normalized to one.

    """

    input_workspace = _workspace_provider.get_workspace_handle(input_workspace)
    assert isinstance(input_workspace, _IMDWorkspace)

    if x is None:
        x = _slice_algorithm.get_available_axis(input_workspace)[0]

    if y is None:
        y = _slice_algorithm.get_available_axis(input_workspace)[1]

    # check to see if x is just a name e.g 'DeltaE' or a full binning spec e.g. 'DeltaE,0,1,100'
    if ',' in x:
        x_axis = _string_to_axis(x)
    else:
        x_axis = Axis(units=x, start=None, end=None, step=None) # The model will fill in the rest

    # check to see if y is just a name e.g 'DeltaE' or a full binning spec e.g. 'DeltaE,0,1,100'
    if ',' in y:
        y_axis = _string_to_axis(y)
    else:
        y_axis = Axis(y, start=None, end=None, step=None) # The model will take care of the missing parameters

    # By this point both x_axis and y_axis should be 'Axis' objects

    # intensity values are set to None since we are not using them. These values are used for plotting/colorbars
    slice_array, extents, colormap, norm = _slice_model.get_slice_plot_data(selected_workspace=input_workspace,
                                                                            x_axis=x_axis, y_axis=y_axis, smoothing=None,
                                                                            intensity_start=None,
                                                                            intensity_end=None,
                                                                            norm_to_one=normalized,
                                                                            colourmap=None)
    if ret_val == 'slice':
        return slice_array
    elif ret_val == 'extents':
        return extents
    elif ret_val == 'both':
        return slice_array, extents
    else:
        raise ValueError("ret_val should be 'slice', 'extents' or 'both' and not %s " % ret_val)

