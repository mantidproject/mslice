from __future__ import (absolute_import, division, print_function)

import mslice.util.mantid.init_mantid # noqa: F401
from mslice.plotting.pyplot import *  # noqa: F401
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mslice.cli.helperfunctions import is_slice, is_cut
from ._mslice_commands import *  # noqa: F401
from mslice.app import is_gui
from mslice.cli.helperfunctions import (_check_workspace_name, _check_workspace_type, _get_overplot_key,
                                        _update_overplot_checklist, _update_legend)
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.plotting.globalfiguremanager import GlobalFigureManager
from mslice.workspace.histogram_workspace import HistogramWorkspace

# This is not compatible with mslice as we use a separate
# global figure manager see _mslice_commands.Show
del show  # noqa: F821


# MSlice Matplotlib Projection
class MSliceAxes(Axes):
    name = 'mslice'

    def errorbar(self, *args, **kwargs):
        from mslice.cli.plotfunctions import errorbar
        if is_cut(*args):
            return errorbar(self, *args, **kwargs)
        else:
            return Axes.errorbar(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        from mslice.cli.plotfunctions import pcolormesh
        if is_slice(*args):
            return pcolormesh(self, *args, **kwargs)
        else:
            return Axes.pcolormesh(self, *args, **kwargs)

    def recoil(self, workspace, element=None, rmm=None):
        from mslice.app.presenters import get_slice_plotter_presenter
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
        from mslice.app.presenters import get_slice_plotter_presenter
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
