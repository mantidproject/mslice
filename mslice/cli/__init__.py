from __future__ import (absolute_import, division, print_function)

from ._mslice_commands import *  # noqa: F401

import mantid.kernel
import mantid.plots.plotfunctions
import mantid.plots.plotfunctions3D
from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mslice.plotting.cli_helperfunctions import validate_args, is_cut
from mslice.workspace.histogram_workspace import HistogramWorkspace


def _check_workspace_name(workspace):
    if isinstance(workspace, Workspace):
        return
    if not isinstance(workspace, str):
        raise TypeError('InputWorkspace must be a workspace or a workspace name')
    if not workspace_exists(workspace):
        raise TypeError('InputWorkspace %s could not be found.' % workspace)


def _check_workspace_type(workspace, correct_type):
    """Check a PSD workspace is MatrixWorkspace, or non-PSD is the specified type"""
    if workspace.is_PSD:
        if isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type - run MakeProjection first.")
        if not isinstance(workspace, correct_type):
            raise RuntimeError("Incorrect workspace type.")
    else:
        if not isinstance(workspace, MatrixWorkspace):
            raise RuntimeError("Incorrect workspace type.")


def PlotSlice(InputWorkspace, IntensityStart="", IntensityEnd="", Colormap=DEFAULT_CMAP):
    """
    Creates mslice standard matplotlib plot of a slice workspace.

    :param InputWorkspace: Workspace to plot. The parameter can be either a python
    handle to the workspace OR the workspace name as a string.
    :param IntensityStart: Lower bound of the intensity axis (colorbar)
    :param IntensityEnd: Upper bound of the intensity axis (colorbar)
    :param Colormap: Colormap name as a string. Default is 'viridis'.
    :return:
    """
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace, HistogramWorkspace)
    slice_presenter = app.MAIN_WINDOW.slice_plotter_presenter
    slice_presenter.change_intensity(workspace.name, IntensityStart, IntensityEnd)
    slice_presenter.change_colourmap(workspace.name, Colormap)
    slice_presenter.plot_from_cache(workspace)


def PlotCut(InputWorkspace, IntensityStart=0, IntensityEnd=0, PlotOver=False):
    """
    Create mslice standard matplotlib plot of a cut workspace.

    :param InputWorkspace: Workspace to cut. The parameter can be either a python handle to the workspace
    OR the workspace name as a string.
    :param IntensityStart: Lower bound of the y axis
    :param IntensityEnd: Upper bound of the y axis
    :param PlotOver: if true the cut will be plotted on an existing figure.
    :return:
    """
    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")
    if IntensityStart == 0 and IntensityEnd == 0:
        intensity_range = None
    else:
        intensity_range = (IntensityStart, IntensityEnd)
    app.MAIN_WINDOW.cut_plotter_presenter.plot_cut_from_workspace(workspace, intensity_range=intensity_range,
                                                                  plot_over=PlotOver)


class MSliceAxes(Axes):
    name = 'mslice'

    def plot(self, *args, **kwargs):

        if validate_args(*args):
            mantid.kernel.logger.debug('using mantid.plots.plotfunctions')
            if is_cut(*args):
                PlotCut(*args, **kwargs)
            else:
                PlotSlice(*args, **kwargs)
        else:
            return Axes.plot(self, *args, **kwargs)


register_projection(MSliceAxes)
