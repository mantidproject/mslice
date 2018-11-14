from __future__ import (absolute_import, division, print_function)

import mantid.kernel

from mslice.cli.cli_helperfunctions import is_slice, is_cut
from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.cli._mslice_commands import *  # noqa: F401

# Imports for mslice projections
from matplotlib.axes import Axes
from matplotlib.projections import register_projection


# Separate presenters for cli
cli_cut_plotter_presenter = CutPlotterPresenter()
cli_slice_plotter_presenter = SlicePlotterPresenter()


# MSlice Matplotlib Projection
class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):
        from mslice.cli.cli_mslice_projection_functions import PlotCutMsliceProjection
        if is_cut(*args):
            try:
                lines = PlotCutMsliceProjection(cut_presenter=cli_cut_plotter_presenter, *args, **kwargs)
                return lines
            except Exception as e:
                print('Mslice Projection Error: ' + repr(e))
        else:
            return Axes.plot(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        from mslice.cli.cli_mslice_projection_functions import PlotSliceMsliceProjection

        if is_slice(*args):
            try:
                mantid.kernel.logger.debug('using mantid.plots.plotfunctions')
                quadmesh = PlotSliceMsliceProjection(slice_presenter=cli_slice_plotter_presenter, *args, **kwargs)
                return quadmesh
            except Exception as e:
                print('MSlice Projection Error: ' + repr(e))
        else:
            return Axes.pcolormesh(self, *args, **kwargs)


register_projection(MSliceAxes)
