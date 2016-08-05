from slice_plotter import SlicePlotter
from mantid.simpleapi import mtd
#from plotting import pyplot as plt
from itertools import chain
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.tri as tri
import time
import matplotlib.pyplot as plt
import numpy as np


def convert_to_centers(bin_limits):
    return [(bin_limits[i]+bin_limits[i+1])/2 for i in range(len(bin_limits)-1)]


def get_aspect_ratio(self, workspace):
        return 1


class MatplotlibSlicePlotter(SlicePlotter):
    def display_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                      colourmap):
        workspace = mtd[selected_workspace]
        blocksize = workspace.blocksize()
        # doing it the wrong way just to check if it is worth it
        xdata = workspace.readX(0)[1:]
        ydata = workspace.readY(0)
        workspace_indices = np.full((blocksize,),0)
        for i in range(1,workspace.getNumberHistograms()):
            ydata = np.append(ydata,workspace.readY(i))
            xdata = np.append(xdata,workspace.readX(i)[1:])
            workspace_indices = np.append(workspace_indices,np.full((blocksize,),i))

        plt.tricontourf(xdata, workspace_indices,
                      ydata)
        plt.show()
       # plt.figure()
       # plt.tripcolor(workspace_indices,xdata,ydata)
