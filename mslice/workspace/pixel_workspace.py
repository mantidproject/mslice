from workspace import Workspace
from histogram_workspace import HistogramWorkspace
from mantid.simpleapi import BinMD


class PixelWorkspace(Workspace):

    def __init__(self, pixel_workspace):
        self.workspace = pixel_workspace
        self.histogram_workspace = None

    def get_histogram_workspace(self):
        if self.histogram_workspace is None:
            dim_values = []
            for x in range(6):
                try:
                    dim = self.workspace.getDimension(x)
                    dim_info = dim.getName() + ',' + str(dim.getMinimum()) + ',' + str(dim.getMaximum()) + ',' + str(100)
                except RuntimeError:
                    dim_info = None
                dim_values.append(dim_info)
            histo_workspace = BinMD(InputWorkspace=self.workspace, OutputWorkspace="",
                                    AlignedDim0=dim_values[0], alignedDim1=dim_values[1],
                                    alignedDim2=dim_values[2], alignedDim3=dim_values[3],
                                    alignedDim4=dim_values[4], alignedDim5=dim_values[5])
            self.histogram_workspace = HistogramWorkspace(histo_workspace)
        return self.histogram_workspace

    def get_signal(self):
        return self.get_histogram_workspace().get_signal()

    def get_error(self):
        return self.get_histogram_workspace().get_error()

    def get_variance(self):
        return self.get_histogram_workspace().get_variance()
