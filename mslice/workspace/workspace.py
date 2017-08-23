from mantid.api import MatrixWorkspace

class Workspace(object):

    def __init__(self, matrix_workspace):
        self.matrix_workspace = matrix_workspace
        mws = MatrixWorkspace() #temp
        #  mws.
        pass

    def get_coordinates(self):
        pass

    def get_signal(self):
        pass

    def get_error(self):
        pass

    def get_variance(self):
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self,other):
        pass

    def __div__(self, other):
        pass

    def __pow__(self, power):
        pass

    def __neg__(self):
        pass
