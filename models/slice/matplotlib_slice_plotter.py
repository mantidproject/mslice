from slice_plotter import SlicePlotter
from models.workspacemanager.mantid_workspace_provider import MantidWorkspaceProvider
from mantid.simpleapi import BinMD
from mantid.api import IMDEventWorkspace
from math import floor
import numpy as np
from plotting import pyplot as plt


def get_aspect_ratio(workspace):
    return 'auto'

def get_number_of_steps(axis):
    return int(max(1, floor(float(axis.end) - float(axis.start))/float(axis.step)))

class MatplotlibSlicePlotter(SlicePlotter):
    def __init__(self):
        self._workspace_provider = MantidWorkspaceProvider()

    def display_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                      colourmap):
        workspace = self._workspace_provider.get_workspace_handle(selected_workspace)
        if isinstance(workspace,IMDEventWorkspace):
            # TODO ask if this slice should live in ADS after plotting?
            # TODO implement axis swapping
            # TODO implement input validation and return appropriate error codes
            # TODO auto_generate input in case of missing input
            # TODO make shown workspaces refresh after this is called
            n_x_bins = get_number_of_steps(x_axis)
            n_y_bins = get_number_of_steps(y_axis)
            x_dim = workspace.getDimension(0)
            y_dim = workspace.getDimension(1)
            xbinning = x_dim.getName() + "," + x_axis.start + "," + x_axis.end + "," + str(n_x_bins)
            ybinning = y_dim.getName() + "," + y_axis.start + "," + y_axis.end + "," + str(n_y_bins)
            slice = BinMD(InputWorkspace=workspace, AxisAligned="1", AlignedDim0=xbinning, AlignedDim1=ybinning)
            plot_data = slice.getSignalArray() / slice.getNumEventsArray()
            plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
            plot_data = np.rot90(plot_data)
            x_step = x_dim.getX(1) - x_dim.getX(0)
            x = np.arange(float(x_axis.start), float(x_axis.end), float(x_axis.step) )
            y_step = y_dim.getX(1) - y_dim.getX(0)
            y = np.arange(float(y_axis.start), float(y_axis.end), float(y_axis.step) )
            #TODO check maths to see if x and y align properly with plot or are off by half bin/ off by one
            xx, yy = np.meshgrid(y, x, indexing='ij')
            plt.pcolormesh(yy, xx, np.flipud(plot_data))
            plt.xlabel(x_dim.getName())
            plt.ylabel(y_dim.getName())
        else:
            ydata = []
            for i in range(workspace.getNumberHistograms()-1,-1,-1):
                ydata.append(workspace.readY(i))
            x_left = workspace.readX(0)[0]
            x_right = workspace.readX(0)[-1]
            y_top = workspace.getNumberHistograms() - 1
            y_bottom = 0
            plt.imshow(ydata,extent=[x_left, x_right, y_bottom, y_top], aspect=get_aspect_ratio(workspace))




