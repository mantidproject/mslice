from __future__ import (absolute_import, division, print_function)

import mantid.kernel

from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mslice.cli.cli_helperfunctions import _check_workspace_type, _check_workspace_name, is_slice, is_cut, is_gui

from mslice.cli._mslice_commands import *  # noqa: F401


# Show function to keep event loop running
def show():
    app.qpp.show()


# MSlice Matplotlib Projection
class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):
        from mslice.cli.cli_mslice_projection_functions import PlotCutMsliceProjection
        if is_cut(*args):
            try:
                lines = PlotCutMsliceProjection(*args, **kwargs)
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
                quadmesh = PlotSliceMsliceProjection(*args, **kwargs)
                return quadmesh
            except Exception as e:
                print('MSlice Projection Error: ' + repr(e))
        else:
            return Axes.pcolormesh(self, *args, **kwargs)


register_projection(MSliceAxes)
