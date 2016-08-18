from cut_plotter import CutPlotter
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from math import floor
from mantid.simpleapi import BinMD
import plotting.pyplot as plt
import numpy as np

def to_float(x):
    if x:
        return float(x)
    else:
        return None


class MatplotlibCutPlotter(CutPlotter):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def plot_cut(self, selected_workspace, cut_axis, integration_start, integration_end,
                 intensity_start, intensity_end, norm_to_one, plot_over):
        #selected_workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        from mantid.simpleapi import AnalysisDataService;selected_workspace = AnalysisDataService[selected_workspace]
        # On merge remove line above and uncomment commented line
        # TODO check if is md workspace
        cut_binning = cut_axis.units + ',' + cut_axis.start + ',' + cut_axis.end + ',' + '100'
        print ('Warning : disregarding input and binning to 100 bins and integrating along Q')
        integration_binning = "|Q|," + integration_start + "," + integration_end +",1"
        cut = BinMD(selected_workspace, AxisAligned1=integration_binning, AlignedDim0=cut_binning)
        with np.errstate(invalid='ignore'):
            plot_data = cut.getSignalArray() / cut.getNumEventsArray
        raise Exception("%s"%plot_data.shape)

    def _get_number_of_steps(self, axis):
        return int(max(1, floor(axis.end - axis.start)/axis.step))