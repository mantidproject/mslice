from __future__ import (absolute_import, division, print_function)

import mslice.util.mantid.init_mantid # noqa: F401
from mslice.plotting.pyplot import *  # noqa: F401
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mslice.cli.helperfunctions import is_slice, is_cut
from ._mslice_commands import *  # noqa: F401

# This is not compatible with mslice as we use a separate
# global figure manager see _mslice_commands.Show
del show  # noqa: F821


# MSlice Matplotlib Projection
class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):
        from mslice.cli.projection_functions import PlotCutMsliceProjection
        if is_cut(*args):
            return PlotCutMsliceProjection(self, *args, **kwargs)
        else:
            return Axes.plot(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        from mslice.cli.projection_functions import PlotSliceMsliceProjection
        if is_slice(*args):
            return PlotSliceMsliceProjection(self, *args, **kwargs)
        else:
            return Axes.pcolormesh(self, *args, **kwargs)


register_projection(MSliceAxes)
