from __future__ import (absolute_import, division, print_function)

from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mslice.cli.cli_helperfunctions import is_slice, is_cut


# MSlice Matplotlib Projection
class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):
        from mslice.cli.cli_mslice_projection_functions import PlotCutMsliceProjection
        if is_cut(*args):
            return PlotCutMsliceProjection(*args, **kwargs)
        else:
            return Axes.plot(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        from mslice.cli.cli_mslice_projection_functions import PlotSliceMsliceProjection
        if is_slice(*args):
            return PlotSliceMsliceProjection(*args, **kwargs)
        else:
            return Axes.pcolormesh(self, *args, **kwargs)


register_projection(MSliceAxes)
