
class AlgWorkspaceOps(object):

    def _get_number_of_steps(self, axis):
        return int(max(1, (axis.end - axis.start)/axis.step))

    def get_axis_range(self, workspace, dimension_name):
        return tuple(self._workspace_provider.get_limits(workspace, dimension_name))

    def set_workspace_provider(self, workspace_provider):
        self._workspace_provider = workspace_provider

    def getComment(self, workspace):
        return self._workspace_provider.getComment(workspace)

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
        dim_names = []
        if isinstance(workspace, str):
            workspace = self._workspace_provider.get_workspace_handle(workspace)
        for i in range(workspace.getNumDims()):
            dim_names.append(workspace.getDimension(i).getName())
        return dim_names
