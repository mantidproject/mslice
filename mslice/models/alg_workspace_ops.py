import numpy as np
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle

def get_number_of_steps(axis):
    return int(np.ceil(max(1, (axis.end - axis.start)/axis.step)))


def get_axis_range(workspace, dimension_name):
    workspace = get_workspace_handle(workspace)
    return tuple(workspace.limits[dimension_name])


def fill_in_missing_input(axis, workspace):
    dim = workspace.getDimensionIndexByName(axis.units)
    dim = workspace.getDimension(dim)

    if axis.start is None:
        axis.start = dim.getMinimum()

    if axis.end is None:
        axis.end = dim.getMaximum()

    if axis.step is None:
        axis.step = (axis.end - axis.start)/100


def get_available_axes(workspace):
    workspace = get_workspace_handle(workspace)
    if not workspace.is_PSD:
        return ['|Q|', 'Degrees', 'DeltaE']
    dim_names = []
    for i in range(workspace.raw_ws.getNumDims()):
        dim_names.append(workspace.raw_ws.getDimension(i).getName())
    return dim_names

def get_other_axis(workspace, axis):
    all_axis = get_available_axes(workspace)
    all_axis.remove(axis.units)
    return all_axis[0]
