import numpy as np
from workspace import Workspace
from mantid.simpleapi import CreateMDHistoWorkspace, ReplicateMD

class HistogramWorkspace(Workspace):


    def __init__(self, histogram_workspace):
        self.inner_workspace = histogram_workspace


    def get_signal(self):
        return self.inner_workspace.getSignalArray()

    def get_error(self):
        return np.sqrt(self.get_variance())

    def get_variance(self):
        return self.inner_workspace.getErrorSquaredArray()

    def _binary_op_array(self, operator, other):
        min = np.amin(other)
        max = np.amax(other)
        size = other.size
        ws = CreateMDHistoWorkspace(Dimensionality=1, Extents='' + str(min) + ',' + str(max),
                                    SignalInput=other, ErrorInput=other, NumberOfBins=str(size),
                                    Names=self.inner_workspace.getDimension(0).getName(), Units='MomentumTransfer')
        try:
            replicated = ReplicateMD(self.inner_workspace, ws)
            return operator(self.inner_workspace, replicated)
        except RuntimeError:
            raise RuntimeError("List or array must have same number of elements as an axis of the workspace")
