import numpy as np
from workspace import Workspace
from mantid.api import IMDHistoWorkspace


class HistogramWorkspace(Workspace):


    def __init__(self, histogram_workspace):
        self.workspace = histogram_workspace

    def get_signal(self):
        return self.workspace.getSignalArray()

    def get_error(self):
        return np.sqrt(self.get_variance())

    def get_variance(self):
        return self.workspace.getErrorSquaredArray()
