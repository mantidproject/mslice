import os.path
import sys

import numpy as np
from mantid.api import MDNormalization
from mantid.kernel import ConfigService
from qtpy.QtWidgets import QFileDialog
from scipy.io import savemat

from mslice.models.axis import Axis
from mslice.models.labels import get_display_name
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.plotting.plot_window.quick_options import QuickError
from mslice.util.mantid.mantid_algorithms import (
    CreateMDHistoWorkspace,
    SaveAscii,
    SaveMD,
    SaveNexus,
    SaveNXSPE,
)
from mslice.workspace.helperfunctions import WrapWorkspaceAttribute
from mslice.workspace.histogram_workspace import HistogramWorkspace


def get_save_directory(
    multiple_files=False,
    save_as_image=False,
    default_ext=None,
    parent=None,
):
    """Show a file dialog so the user can choose where to save.

    Returns:
        A tuple of (directory, filename, extension).
        If the save dialog is cancelled, returns (None, None, None).
    """
    options = (
        QFileDialog.Option.DontUseNativeDialog
        if sys.platform.startswith("linux")
        else QFileDialog.Option(0)
    )
    if multiple_files:
        directory = QFileDialog.getExistingDirectory(parent, options=options)
        if not directory:
            return None, None, None
        return directory, None, default_ext

    filter_specs = [
        ("Nexus (*.nxs)", ".nxs"),
        ("NXSPE (*.nxspe)", ".nxspe"),
        ("Ascii (*.txt)", ".txt"),
        ("Matlab (*.mat)", ".mat"),
    ]
    if save_as_image:
        filter_specs = [
            ("Image (*.png)", ".png"),
            ("PDF (*.pdf)", ".pdf"),
            *filter_specs,
        ]

    filter_to_ext = dict(filter_specs)
    ext_to_filter = {ext.casefold(): name for name, ext in filter_specs}

    # using qt native dialog as there is a bug with gtk on linux that causes crashes when changing filter value and modifying file name
    file_dialog = QFileDialog(parent, options=options)
    file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    file_dialog.setNameFilters([name for name, _ in filter_specs])

    if default_ext:
        requested_ext = str(default_ext).casefold()
        if not requested_ext.startswith("."):
            requested_ext = "." + requested_ext
        selected_filter = ext_to_filter.get(requested_ext)
        if selected_filter is None:
            raise ValueError(f"Unsupported default extension: {default_ext!r}")
        file_dialog.selectNameFilter(selected_filter)

    if not file_dialog.exec():
        return None, None, None

    selected_files = file_dialog.selectedFiles()
    if not selected_files:
        return None, None, None

    path = selected_files[0]
    directory = os.path.dirname(path)
    filename = os.path.basename(path)

    selected_ext = filter_to_ext.get(file_dialog.selectedNameFilter())
    suffix = os.path.splitext(filename)[1]
    if suffix in ("", ".") and selected_ext:
        filename = filename.rstrip(".") + selected_ext

    extension = os.path.splitext(filename)[1]
    return directory, filename, extension


def save_nexus(workspace, path):
    if isinstance(workspace, HistogramWorkspace):
        _save_histogram_workspace(workspace, path)
        return

    loader_name = get_workspace_handle(workspace).loader_name()
    if loader_name is not None and loader_name == "LoadNXSPE":
        raise RuntimeError(
            "An NXSPE file cannot be saved as a Nexus - metadata may be lost."
        )

    with WrapWorkspaceAttribute(workspace):
        SaveNexus(InputWorkspace=workspace, Filename=path)


def save_nxspe(workspace, path):
    if isinstance(workspace, HistogramWorkspace):
        _save_histogram_workspace(workspace, path)
        return

    loader_name = get_workspace_handle(workspace).loader_name()
    if loader_name is not None and loader_name != "LoadNXSPE":
        raise RuntimeError(
            "A Nexus cannot be saved as an NXSPE file - metadata may be lost."
        )

    if len(workspace.axes) == 2:
        _validate_nxspe_save(workspace=workspace)
        return

    with WrapWorkspaceAttribute(workspace):
        SaveNXSPE(InputWorkspace=workspace, Filename=path)


def save_ascii(workspace, path):
    if workspace.is_slice:
        _save_slice_to_ascii(workspace, path)
    else:
        if isinstance(workspace, HistogramWorkspace):
            _save_cut_to_ascii(workspace, path)
        else:
            SaveAscii(InputWorkspace=workspace, Filename=path, WriteSpectrumID=False)


def save_matlab(workspace, path):
    labels = {}
    if isinstance(workspace, HistogramWorkspace):
        if workspace.is_slice:
            x, y, e = _get_slice_mdhisto_xye(workspace.raw_ws)
            labels = {
                "x": get_display_name(workspace.axes[0]),
                "y": get_display_name(workspace.axes[1]),
            }
        else:
            x, y, e = _get_md_histo_xye(workspace.raw_ws)
            labels = {"x": get_display_name(workspace.axes[0])}
    else:
        if not hasattr(workspace.raw_ws, "extractX"):
            raise RuntimeError("Only cuts and slices can be saved as Matlab.")

        if workspace.is_slice:
            x = []
            for dim in [workspace.raw_ws.getDimension(i) for i in range(2)]:
                x.append(
                    np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins())
                )
            x = np.array(x, dtype=object)
            # We're saving a 2D RebinnedWorkspace which always has DeltaE along x
            ix = next(i for i, ax in enumerate(workspace.axes) if "DeltaE" in ax.units)
            iy = 0 if ix == 1 else 1
            labels = {
                "x": get_display_name(workspace.axes[ix]),
                "y": get_display_name(workspace.axes[iy]),
            }
        else:
            x = workspace.raw_ws.extractX()
        y = workspace.raw_ws.extractY()
        e = workspace.raw_ws.extractE()
    mdict = {"x": x, "y": y, "e": e, "labels": labels}
    savemat(_to_absolute_path(path), mdict=mdict)


def _save_histogram_workspace(workspace, path):
    if workspace.is_slice:
        workspace = get_workspace_handle(workspace.name[2:])

    with WrapWorkspaceAttribute(workspace):
        SaveMD(InputWorkspace=workspace, Filename=path)


def _save_cut_to_ascii(workspace, output_path):
    # get integration ranges from the name
    cut_axis, integration_axis = tuple(workspace.axes)

    x, y, e = _get_md_histo_xye(workspace.raw_ws)
    header = f'MSlice Cut of workspace "{workspace.parent}"\n'
    header += f"Cut axis: {cut_axis}\n"
    header += f"Integration axis: {integration_axis}\n"
    header += f"({get_display_name(cut_axis)}) (Signal) (Error)"
    out_data = np.c_[x, y, e]
    _output_data_to_ascii(output_path, out_data, header)


def _save_slice_to_ascii(workspace, output_path):
    if isinstance(workspace, HistogramWorkspace):
        x, y, e = _get_slice_mdhisto_xye(workspace.raw_ws)
        ix, iy = (0, 1)
    else:
        x = []
        for dim in [workspace.raw_ws.getDimension(i) for i in range(2)]:
            x.append(
                np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins())[
                    np.newaxis
                ]
            )
        x[1] = x[1].T
        y = workspace.raw_ws.extractY()
        e = workspace.raw_ws.extractE()
        e_axes = [i for i, ax in enumerate(workspace.axes) if "DeltaE" in ax.units]
        assert len(e_axes) > 0
        ix = e_axes[0]
        iy = 0 if ix == 1 else 1
    dim_sz = [
        workspace.raw_ws.getDimension(i).getNBins()
        for i in range(workspace.raw_ws.getNumDims())
    ]
    nbins = np.prod(dim_sz)
    x = [np.reshape(x0, nbins) for x0 in np.broadcast_arrays(*x)]
    y = np.reshape(y, nbins)
    e = np.reshape(e, nbins)
    out_data = np.column_stack(tuple(x + [y, e]))
    labels = {
        "x": get_display_name(workspace.axes[ix]),
        "y": get_display_name(workspace.axes[iy]),
    }
    header = f'MSlice Slice of workspace "{workspace.name}"'
    header += "\n({}) ({}) (Signal) (Error)".format(labels["x"], labels["y"])
    _output_data_to_ascii(output_path, out_data, header)


def load_from_ascii(file_path, ws_name):
    file = open(file_path, "r")  # noqa: SIM115
    line = file.readline()
    header = ""
    while line.startswith("#"):
        header += line
        line = file.readline()
    cut_axis_str = header[header.find("Cut axis: ") + 10 :]
    cut_axis = Axis(*cut_axis_str[: cut_axis_str.find("\n")].split(","))
    integration_axis_str = header[header.find("Integration axis: ") + 17 :]
    integration_axis = Axis(
        *integration_axis_str[: integration_axis_str.find("\n")].split(",")
    )
    if not header.startswith("# MSlice Cut"):
        raise ValueError
    x, y, e = np.loadtxt(file).transpose()
    extents = str(np.min(x)) + "," + str(np.max(x))
    nbins = len(x)
    ws_out = CreateMDHistoWorkspace(
        OutputWorkspace=ws_name,
        SignalInput=y,
        ErrorInput=e,
        Dimensionality=1,
        Extents=extents,
        NumberOfBins=nbins,
        Names="Dim1",
        Units=cut_axis.units,
    )
    ws_out.axes = [cut_axis, integration_axis]


def _get_md_histo_xye(histo_ws):
    dim = histo_ws.getDimension(0)
    if dim.getNBins() == 1:
        dim = histo_ws.getDimension(1)
    start = dim.getMinimum()
    end = dim.getMaximum()
    nbin = dim.getNBins()
    x = np.linspace(start, end, nbin)
    y = np.squeeze(histo_ws.getSignalArray())
    e = np.squeeze(np.sqrt(histo_ws.getErrorSquaredArray()))
    if histo_ws.displayNormalization() == MDNormalization.NumEventsNormalization:
        num_events = histo_ws.getNumEventsArray()
        y = y / num_events
        e = e / num_events
    return x, y, e


def _get_slice_mdhisto_xye(histo_ws):
    dim_sz = [histo_ws.getDimension(i).getNBins() for i in range(histo_ws.getNumDims())]
    nz_dim = [i for i, v in enumerate(dim_sz) if v > 1]
    numdim = len(nz_dim)
    x = []
    for i in nz_dim:
        dim = histo_ws.getDimension(i)
        start = dim.getMinimum()
        end = dim.getMaximum()
        nshape = [1] * numdim
        nshape[i] = dim_sz[i]
        x.append(np.reshape(np.linspace(start, end, dim_sz[i]), tuple(nshape)))
    y = histo_ws.getSignalArray()
    e = np.sqrt(histo_ws.getErrorSquaredArray())
    if histo_ws.displayNormalization() == MDNormalization.NumEventsNormalization:
        num_events = histo_ws.getNumEventsArray()
        y = y / num_events
        e = e / num_events
    return x, y, e


def _output_data_to_ascii(output_path, out_data, header):
    try:
        np.savetxt(
            _to_absolute_path(str(output_path)), out_data, fmt="%12.9e", header=header
        )
    except OSError as e:
        raise RuntimeError(f"Error occurred: {e}")


def _to_absolute_path(filepath: str) -> str:
    """Returns an absolute path in which a file should be saved."""
    if os.path.isabs(filepath):
        return filepath
    return os.path.join(ConfigService.getString("defaultsave.directory"), filepath)


def _validate_nxspe_save(workspace):
    """Validates if the nxspe file can be saved."""
    q_check = np.allclose(
        [workspace.axes[0].start, workspace.axes[0].end],
        [workspace.limits["|Q|"][0], workspace.limits["|Q|"][1]],
        atol=1e-4,
        rtol=0,
    )
    e_check = np.allclose(
        [workspace.axes[1].start, workspace.axes[1].end],
        [workspace.limits["DeltaE"][0], workspace.limits["DeltaE"][1]],
        atol=1e-4,
        rtol=0,
    )
    if not (q_check and e_check):
        QuickError(
            "Cannot save a NXSPE file because the workspace limits and the axis limits don't match."
        )
