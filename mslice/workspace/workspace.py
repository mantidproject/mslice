import numpy as np
import operator
from mantid.simpleapi import CreateWorkspace


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
        if isinstance(other, self.__class__):
            if self.check_dimensions(other):
                inner_res = operator(self.inner_workspace, other.inner_workspace)
            else:
                raise RuntimeError("workspaces must have same dimensionality for binary operations (+, -, *, /)")
        elif isinstance(other, np.ndarray):
            inner_res = self._binary_op_array(operator, other)
        else:
            inner_res = operator(self.inner_workspace, other)
        workspace_type = type(self)
        return workspace_type(inner_res)

    def _binary_op_array(self, operator, other):
        if other.size == self.get_signal().size:
            new_signal = operator(other, self.get_signal()[0])
            return CreateWorkspace(self.inner_workspace.extractX(), new_signal,
                                   self.inner_workspace.extractE(), outputWorkspace=str(self))
        else:
            raise RuntimeError("List or array must have same number of elements as an axis of the workspace")

    def check_dimensions(self, workspace_to_check):
        for i in range(self.inner_workspace.getNumDims()):
            if self.inner_workspace.getDimension(i).getNBins() != \
                    workspace_to_check.inner_workspace.getDimension(i).getNBins():
                return False
        return True

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
