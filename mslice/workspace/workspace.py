from mantid.api import MatrixWorkspace
import numpy as np

class Workspace(object):

    def __init__(self, matrix_workspace):
        self.workspace = matrix_workspace
        # print self.matrix_workspace.readY(4)
        # x = self + self
        # print x.readY(4)

    def get_workspace(self):
        return self.workspace

    def get_coordinates(self):
        number_of_dimensions = self.workspace.getNumDims()
        coords = {}
        for i in range(number_of_dimensions):
            dim = self.workspace.getDimension(i)
            coords[dim.getName()] = np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins())
        return coords

    def get_signal(self):
        return self.workspace.extractY()

    def get_error(self):
        return self.workspace.extractE()

    def get_variance(self):
        np.square(self.get_error())

    def __add__(self, other):
        if isinstance(other, Workspace):
            return self.workspace + other.get_workspace() #other way to do this?
        else:
            return self.workspace + other

    def __sub__(self, other):
        return self.workspace - other

    def __mul__(self,other):
        return self.workspace * other

    def __div__(self, other):
        return self.workspace / other

    def __pow__(self, power):
        pass

    def __neg__(self):
        pass
