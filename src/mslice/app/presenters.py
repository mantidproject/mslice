from mslice.presenters.cut_plotter_presenter import CutPlotterPresenter
from mslice.presenters.slice_plotter_presenter import SlicePlotterPresenter
from mslice.cli.views.cli_data_loader import CLIDataLoaderWidget
from mslice.presenters.data_loader_presenter import DataLoaderPresenter
from mslice.presenters.powder_projection_presenter import PowderProjectionPresenter
from mslice.cli.views.cli_powder import CLIPowderWidget
from mslice.models.projection.powder.mantid_projection_calculator import (
    MantidProjectionCalculator,
)
from . import is_gui
from mslice import app

# Separate presenters for the CLI
cli_cut_plotter_presenter = CutPlotterPresenter()
cli_slice_plotter_presenter = SlicePlotterPresenter()
cli_dataloader_presenter = DataLoaderPresenter(CLIDataLoaderWidget())
cli_powder_presenter = PowderProjectionPresenter(
    CLIPowderWidget(), MantidProjectionCalculator()
)


def get_dataloader_presenter():
    if is_gui():
        return app.MAIN_WINDOW.dataloader_presenter
    else:
        return cli_dataloader_presenter


def get_slice_plotter_presenter():
    if is_gui():
        return app.MAIN_WINDOW.slice_plotter_presenter
    else:
        return cli_slice_plotter_presenter


def get_cut_plotter_presenter():
    if is_gui():
        return app.MAIN_WINDOW.cut_plotter_presenter
    else:
        return cli_cut_plotter_presenter


def get_powder_presenter():
    if is_gui():
        return app.MAIN_WINDOW.powder_presenter
    else:
        return cli_powder_presenter
