from __future__ import (absolute_import, division, print_function)

import mslice.util.mantid.init_mantid # noqa: F401
from mslice.plotting.pyplot import *  # noqa: F401
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mslice.cli.helperfunctions import is_slice, is_cut, HistogramWorkspace, _intensity_to_action, _function_to_intensity
from ._mslice_commands import *  # noqa: F401
from mslice.app import is_gui
from mslice.cli.helperfunctions import (_string_to_integration_axis, _process_axis, _check_workspace_name,
                                        _check_workspace_type, _get_overplot_key, _overplot_keys,
                                        _update_overplot_checklist, _update_legend)


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

    def recoil(self, workspace, element=None, rmm=None):
        _check_workspace_name(workspace)
        workspace = get_workspace_handle(workspace)
        _check_workspace_type(workspace, HistogramWorkspace)

        key = _get_overplot_key(element, rmm)

        if rmm is not None:
            plot_handler = GlobalFigureManager.get_active_figure()._plot_handler
            plot_handler._arb_nuclei_rmm = rmm

        get_slice_plotter_presenter().add_overplot_line(workspace.name, key, recoil=True, cif=None)

        _update_overplot_checklist(key)
        _update_legend()

    def bragg(self, workspace, element=None, cif=None):
        _check_workspace_name(workspace)
        workspace = get_workspace_handle(workspace)
        _check_workspace_type(workspace, HistogramWorkspace)

        key = _get_overplot_key(element, rmm=None)

        get_slice_plotter_presenter().add_overplot_line(workspace.name, key, recoil=False, cif=cif)
        _update_overplot_checklist(key)
        _update_legend()

    def grid(self, b=None, which='major', axis='both', **kwargs):
        Axes.grid(self, b, which, axis, **kwargs)

        plot_handler = GlobalFigureManager.get_active_figure()._plot_handler
        if plot_handler is not None and not is_gui():
            if axis == 'x':
                plot_handler.manager._xgrid = b
            elif axis == 'y':
                plot_handler.manager._ygrid = b


register_projection(MSliceAxes)
