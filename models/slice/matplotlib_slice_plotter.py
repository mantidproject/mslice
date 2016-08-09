from slice_plotter import SlicePlotter
from mantid.simpleapi import mtd
from plotting import pyplot as plt
import numpy as np
import matplotlib


def get_aspect_ratio(self, workspace):
        return 'auto'


class MatplotlibSlicePlotter(SlicePlotter):
    def display_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                      colourmap):
        workspace = mtd[selected_workspace]
        plot_method = 'imshow'

        if plot_method == 'imshow':
            ydata = []
            for i in range(workspace.getNumberHistograms()-1,-1,-1):
                ydata.append(workspace.readY(i))
            plt.imshow(ydata)

        if plot_method == 'tripcolor':
            blocksize = workspace.blocksize()
            xdata = workspace.readX(0)[1:]
            ydata = workspace.readY(0)
            workspace_indices = np.full((blocksize,),0)
            for i in range(1,workspace.getNumberHistograms()):
                ydata = np.append(ydata,workspace.readY(i))
                xdata = np.append(xdata,workspace.readX(i)[1:])
                workspace_indices = np.append(workspace_indices,np.full((blocksize,),i))

            plt.tripcolor(xdata, workspace_indices,
                          ydata)

        elif plot_method == 'tricontourf':
            blocksize = workspace.blocksize()
            xdata = workspace.readX(0)[1:]
            ydata = workspace.readY(0)
            workspace_indices = np.full((blocksize,),0)
            for i in range(1,workspace.getNumberHistograms()):
                ydata = np.append(ydata,workspace.readY(i))
                xdata = np.append(xdata,workspace.readX(i)[1:])
                workspace_indices = np.append(workspace_indices,np.full((blocksize,),i))

            plt.tricontourf(xdata, workspace_indices,
                            ydata)
