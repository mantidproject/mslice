# Helper tools
from presenters.slice_plotter_presenter import Axis


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

# Projections
from models.projection.powder.mantid_projection_calculator import MantidProjectionCalculator
from mantid.api import Workspace
_powder_projection_model = MantidProjectionCalculator()

def get_projection(input_workspace, axis1, axis2):
    if isinstance(input_workspace, Workspace):
        input_workspace = input_workspace.getName()
    return _powder_projection_model.calculate_projection(input_workspace=input_workspace, axis1=axis1, axis2=axis2)

#Slicing
from models.slice.matplotlib_slice_plotter import MatplotlibSlicePlotter
from models.slice.mantid_slice_algorithm import MantidSliceAlgorithm
_slice_model = MatplotlibSlicePlotter(MantidSliceAlgorithm())


def get_slice(input_workspace, x, y, intensity_min=None, intensity_max=None, normalized=False):
    x_axis = _string_to_axis(x)
    y_axis = _string_to_axis(y)
    if isinstance(input_workspace, Workspace):
        input_workspace = input_workspace.getName()
    slice_array, extents, colormap, norm = _slice_model.get_slice_plot_data(selected_workspace=input_workspace,
                                                                            x_axis=x_axis, y_axis=y_axis, smoothing=None,
                                                                            intensity_start=intensity_min,
                                                                            intensity_end=intensity_max,
                                                                            norm_to_one=normalized,
                                                                            colourmap=None)
    return slice_array, extents