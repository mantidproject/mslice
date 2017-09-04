import numpy as np
import operator


class Workspace(object):

    def __init__(self, matrix_workspace):
        self.inner_workspace = matrix_workspace

    def get_coordinates(self):
        coords = {}
        for i in range(self.inner_workspace.getNumDims()):
            dim = self.inner_workspace.getDimension(i)
            coords[dim.getName()] = np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins())
        return coords

    def get_signal(self):
        return self.inner_workspace.extractY()

    def get_error(self):
        return self.inner_workspace.extractE()

    def get_variance(self):
        return np.square(self.get_error())

    def _binary_op(self, operator, other):
        if isinstance(other, list):
            other = np.asarray(other)
        if isinstance(other, Workspace):
            inner_res = operator(self.inner_workspace, other.inner_workspace)  # type/dimensionality/binning checks?
        elif isinstance(other, np.ndarray):
            inner_res = self._binary_op_array(operator, other)
        else:
            inner_res = operator(self.inner_workspace, other)
        workspace_type = type(self)
        return workspace_type(inner_res)

    def __add__(self, other):
        return self._binary_op(operator.add, other)

    def __sub__(self, other):
        return self._binary_op(operator.sub, other)

    def __mul__(self, other):
        return self._binary_op(operator.mul, other)

    def __div__(self, other):
        return self._binary_op(operator.div, other)

    def __pow__(self, other):
        new = self
        while other > 1:
            new = new * self
            other -= 1
        return new

    def __neg__(self):
        return self * -1
