import numpy as np
from workspace import Workspace


class HistogramWorkspace(Workspace):


    def __init__(self, histogram_workspace):
        self.inner_workspace = histogram_workspace


    def get_signal(self):
        return self.inner_workspace.getSignalArray()

    def get_error(self):
        return np.sqrt(self.get_variance())

    def get_variance(self):
        return self.inner_workspace.getErrorSquaredArray()
