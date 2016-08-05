from slice_plotter import SlicePlotter
from mantid.simpleapi import mtd
from
class MatplotlibSlicePlotter(SlicePlotter):
    def display_slice(self, selected_workspace, x_axis, y_axis, smoothing, intensity_start, intensity_end, norm_to_one,
                      colourmap):
        workspace = mtd[selected_workspace]