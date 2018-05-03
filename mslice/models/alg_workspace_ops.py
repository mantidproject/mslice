import numpy as np
from mslice.models.workspacemanager.workspace_algorithms import get_comment
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle


class AlgWorkspaceOps(object):

    def _get_number_of_steps(self, axis):
        return int(np.ceil(max(1, (axis.end - axis.start)/axis.step)))

    def get_axis_range(self, workspace, dimension_name):
        workspace = get_workspace_handle(workspace)
        return tuple(workspace.limits[dimension_name])

    def get_comment(self, workspace):
        return get_comment(workspace)

    def _fill_in_missing_input(self,axis,workspace):
        dim = workspace.getDimensionIndexByName(axis.units)
        dim = workspace.getDimension(dim)

        if axis.start is None:
            axis.start = dim.getMinimum()

        if axis.end is None:
            axis.end = dim.getMaximum()

        if axis.step is None:
            axis.step = (axis.end - axis.start)/100

    def get_available_axis(self, workspace):
        workspace = get_workspace_handle(workspace)
        if not workspace.is_PSD:
            return ['|Q|', 'Degrees', 'DeltaE']
        dim_names = []
        for i in range(workspace.raw_ws.getNumDims()):
            dim_names.append(workspace.raw_ws.getDimension(i).getName())
        return dim_names
