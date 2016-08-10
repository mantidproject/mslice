from slice_plotter import SlicePlotter
from mantid.simpleapi import AnalysisDataService
from plotting import pyplot as plt


def get_aspect_ratio(workspace):
    return 'auto'


class MatplotlibSlicePlotter(SlicePlotter):
    def display_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                      colourmap):
        workspace = AnalysisDataService[selected_workspace]
        ydata = []
        for i in range(workspace.getNumberHistograms()-1,-1,-1):
            ydata.append(workspace.readY(i))
        x_left = workspace.readX(0)[0]
        x_right = workspace.readX(0)[-1]
        y_top = workspace.getNumberHistograms() - 1
        y_bottom = 0
        plt.imshow(ydata,extent=[x_left, x_right, y_bottom, y_top], aspect=get_aspect_ratio(workspace))




