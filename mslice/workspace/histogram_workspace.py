import numpy as np
from workspace import Workspace


class HistogramWorkspace(Workspace):

    def __init__(self, histogram_workspace):
        self.histogram_workspace = histogram_workspace

    def get_coordinates(self):
        nevents = self.histogram_workspace.getNumEventsArray()
        return {x: (self.histogram_workspace.signalAt(x) / nevents[x])[0] for x in range(self.histogram_workspace.getNPoints())}
        # ??? not sure of requirements here

    def get_signal(self):
        return self.histogram_workspace.getSignalArray()

    def get_error(self):
        return np.sqrt(self.get_variance())

    def get_variance(self):
        return self.histogram_workspace.getErrorSquaredArray()
