from __future__ import (absolute_import, division, print_function)

from mslice.models.cmap import DEFAULT_CMAP
import mslice.app as app
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.cli.cli_helperfunctions import _check_workspace_type, _check_workspace_name
from mslice.workspace.histogram_workspace import HistogramWorkspace

from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.cli.cli_data_loader import CLIDataLoaderWidget
from mslice.presenters.data_loader_presenter import DataLoaderPresenter

# Separate presenters for cli
cli_cut_plotter_presenter = CutPlotterPresenter()
cli_slice_plotter_presenter = SlicePlotterPresenter()
cli_data_loader_presenter = DataLoaderPresenter(CLIDataLoaderWidget())

#cli_powder_presenter = PowderProjectionPresenter(CLIPowderWidget(), ProjectionCalculator())

# Define cli presenter
cli_presenter = {'cli_cut_plotter_presenter': cli_cut_plotter_presenter,
                 'cli_slice_plotter_presenter': cli_slice_plotter_presenter,
                 'cli_data_loader_presenter': cli_data_loader_presenter,
                 }


def PlotCutMsliceProjection(InputWorkspace, IntensityStart=0, IntensityEnd=0, PlotOver=False):
    """
    Same as the cli PlotCut but returns the relevant axes object.
    """

    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    if not isinstance(workspace, HistogramWorkspace):
        raise RuntimeError("Incorrect workspace type.")
    if IntensityStart == 0 and IntensityEnd == 0:
        intensity_range = None
    else:
        intensity_range = (IntensityStart, IntensityEnd)
    lines = cli_presenter['cli_cut_plotter_presenter'].plot_cut_from_workspace(workspace,
                                                                               intensity_range=intensity_range,
                                                                               plot_over=PlotOver)

    return lines


def PlotSliceMsliceProjection(InputWorkspace, IntensityStart="", IntensityEnd="", Colormap=DEFAULT_CMAP):
    """
    Same as the CLI PlotSlice but returns the relevant axes object.
    """

    _check_workspace_name(InputWorkspace)
    workspace = get_workspace_handle(InputWorkspace)
    _check_workspace_type(workspace, HistogramWorkspace)

    # slice cache needed from main slice plotter presenter
    cli_presenter['cli_slice_plotter_presenter']._slice_cache = app.MAIN_WINDOW.slice_plotter_presenter._slice_cache
    cli_presenter['cli_slice_plotter_presenter'].change_intensity(workspace.name, IntensityStart, IntensityEnd)
    cli_presenter['cli_slice_plotter_presenter'].change_colourmap(workspace.name, Colormap)
    quadmesh = cli_presenter['cli_slice_plotter_presenter'].plot_from_cache(workspace)

    return quadmesh
