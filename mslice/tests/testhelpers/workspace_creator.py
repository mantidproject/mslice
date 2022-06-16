import numpy as np

from mantid.simpleapi import AddSampleLog, CreateMDHistoWorkspace, CreateMDWorkspace

from mslice.workspace.pixel_workspace import PixelWorkspace
from mslice.util.mantid.mantid_algorithms import CreateSampleWorkspace, CreateSimulationWorkspace


def _create_extents(n_dims: int) -> str:
    """Create test data for the extents of a workspace."""
    return ",".join(["-10,10"] * n_dims)


def _create_dim_names(n_dims: int) -> str:
    """Create test data for the dimension names of a workspace."""
    return ",".join(["|Q|", "DeltaE", "C"][:n_dims])


def _create_units(n_dims: int) -> str:
    """Create test data for the dimension units of a workspace."""
    return ",".join(["U"] * n_dims)


def _create_num_of_bins(n_dims: int) -> str:
    """Create test data for the number of bins in each workspace dimension."""
    return ",".join(["10"] * n_dims)


def create_simulation_workspace(e_mode: str, output_name: str, psd: bool = False):
    """Creates a basic simulation workspace for testing purposes."""
    if e_mode == "Direct":
        workspace = CreateSimulationWorkspace(Instrument="MAR", BinParams=[-15, 1, 15], UnitX="DeltaE",
                                              OutputWorkspace=output_name)
    else:
        workspace = CreateSimulationWorkspace(Instrument="OSIRIS", BinParams=[-15, 1, 15], UnitX="DeltaE",
                                              OutputWorkspace=output_name)
    AddSampleLog(workspace=workspace.raw_ws, LogName='Ei', LogText='3.', LogType='Number', StoreInADS=False)

    sim_scattering_data = np.arange(0, 1.5, 0.002).reshape(30, 25).transpose()
    for i in range(25):
        workspace.raw_ws.setY(i, sim_scattering_data[i])
    workspace.e_mode = e_mode
    workspace.e_fixed = 1.1
    workspace.is_PSD = psd
    return workspace


def create_workspace(output_name: str):
    """Creates a basic MSlice Workspace for testing purposes."""
    workspace = CreateSampleWorkspace(NumBanks=1, BankPixelWidth=5, XMin=0.1, XMax=3.1, BinWidth=0.1, XUnit="DeltaE",
                                      OutputWorkspace=output_name)
    return workspace


def create_md_histo_workspace(n_dims: int, output_name: str):
    """Creates a basic MDHistoWorkspace for testing purposes."""
    md_histo_ws = CreateMDHistoWorkspace(Dimensionality=n_dims, Extents=_create_extents(n_dims),
                                         SignalInput=list(range(0, 100)), ErrorInput=list(range(0, 100)),
                                         NumberOfBins=_create_num_of_bins(n_dims), Names=_create_dim_names(n_dims),
                                         Units=_create_units(n_dims), OutputWorkspace=output_name)
    return md_histo_ws


def create_md_workspace(n_dims: int, output_name: str):
    """Creates a basic MDWorkspace for testing purposes."""
    md_ws = CreateMDWorkspace(Dimensions=n_dims, Extents=_create_extents(n_dims), Names=_create_dim_names(n_dims),
                              Units=_create_units(n_dims), OutputWorkspace=output_name)
    return md_ws


def create_pixel_workspace(n_dims: int, output_name: str) -> PixelWorkspace:
    """Creates a basic PixelWorkspace for testing purposes."""
    md_workspace = CreateMDWorkspace(Dimensions=n_dims, Extents=_create_extents(n_dims), Units=_create_units(n_dims),
                                     Names=_create_dim_names(n_dims), OutputWorkspace=output_name)
    return PixelWorkspace(md_workspace, output_name)
