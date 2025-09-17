import numpy as np
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle


def get_number_of_steps(axis):
    if axis.step == 0:
        return 1

    step_num = max(1, (axis.end - axis.start) / axis.step)
    step_num_round = np.around(step_num)
    if np.isclose(step_num_round, step_num, atol=0.05):
        return int(step_num_round)

    return int(np.ceil(step_num))

def get_range_end(start, end, width):
    return min(start + width, end) if width is not None else end

def get_axis_range(workspace, dimension_name):
    workspace = get_workspace_handle(workspace)
    return tuple(workspace.limits[dimension_name])


def get_axis_step(workspace, dimension_name: str) -> float:
    axis_range = get_axis_range(workspace, dimension_name)
    assert (
        len(axis_range) == 3
    )  # Assert that the axis_range tuple is the length we expect
    return axis_range[2]


def fill_in_missing_input(axis, workspace):
    dim = workspace.getDimensionIndexByName(axis.units)
    dim = workspace.getDimension(dim)

    if axis.start is None:
        axis.start = dim.getMinimum()

    if axis.end is None:
        axis.end = dim.getMaximum()

    if axis.step is None:
        axis.step = (axis.end - axis.start) / 100


def get_available_axes(workspace):
    workspace = get_workspace_handle(workspace)
    if not workspace.is_PSD:
        return ["|Q|", "2Theta", "DeltaE"]
    dim_names = []
    for i in range(workspace.raw_ws.getNumDims()):
        dim_names.append(workspace.raw_ws.getDimension(i).name)
        if "Degrees" in dim_names:
            dim_names.remove("Degrees")
            dim_names.append("2Theta")
    return dim_names


def get_other_axis(workspace, axis):
    all_axis = get_available_axes(workspace)
    all_axis.remove(axis.units)
    return all_axis[0]
