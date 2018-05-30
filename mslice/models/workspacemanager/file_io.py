from __future__ import (absolute_import, division, print_function)
import os.path
from mantid.api import MDNormalization
from mslice.util.qt.QtWidgets import QFileDialog
from mslice.models.workspacemanager.workspace_algorithms import run_algorithm
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.workspace.histogram_workspace import HistogramWorkspace

import numpy as np
from scipy.io import savemat


def get_save_directory(multiple_files=False, save_as_image=False, default_ext=None):
    '''
    Show file dialog so user can choose where to save.
    :param multiple_files: boolean - whether more than one file is being saved
    :param save_as_image: boolean - whether to allow saving as a .png/pdf
    :param default_ext: file extension that is selected by default
    :return: path to save directory, name to save the file as, file format extension
    :raises: RuntimeError if dialog is cancelled
    '''
    if multiple_files:
        return QFileDialog.getExistingDirectory(), None, default_ext
    else:
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        filter = "Nexus (*.nxs);; Ascii (*.txt);; Matlab (*.mat)"
        if save_as_image:
            filter += ";; Image (*.png);; PDF (*.pdf)"
        file_dialog.setNameFilter(filter)

        if default_ext:
            ext_to_qtfilter = {'.nxs': 'Nexus (*.nxs)', '.txt': 'Ascii (*.txt)', '.mat': 'Matlab (*.mat)', }
            file_dialog.selectNameFilter(ext_to_qtfilter[default_ext])
        if (file_dialog.exec_()):
            path = str(file_dialog.selectedFiles()[0])
            if '.' not in path: # add extension unless there's one in the name
                path += file_dialog.selectedFilter()[-5:-1]
            ext = path[path.rfind('.'):]
            return os.path.dirname(path), os.path.basename(path), ext
        else:
            raise RuntimeError("dialog cancelled")


def save_nexus(workspace, path, is_slice):
    if isinstance(workspace, HistogramWorkspace):
        if is_slice:
            run_algorithm('SaveMD', store=False, InputWorkspace=get_workspace_handle(workspace.name[2:]), Filename=path)
        else:
            run_algorithm('SaveMD', store=False, InputWorkspace=workspace, Filename=path)
    else:
        run_algorithm('SaveNexus', store=False, InputWorkspace=workspace, Filename=path)


def save_ascii(workspace, path, is_slice):
    if isinstance(workspace, HistogramWorkspace):
        if is_slice:
            _save_slice_to_ascii(workspace, path)
        else:
            _save_cut_to_ascii(workspace, workspace.name, path)
    else:
        run_algorithm('SaveAscii', InputWorkspace=workspace, Filename=path)


def save_matlab(workspace, path, is_slice):
    if isinstance(workspace, HistogramWorkspace):
        if is_slice:
            x, y, e = _get_slice_mdhisto_xye(workspace.raw_ws)
        else:
            x, y, e = _get_md_histo_xye(workspace.raw_ws)
    else:
        if is_slice:
            x = []
            for dim in [workspace.raw_ws.getDimension(i) for i in range(2)]:
                x.append(np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins()))
        else:
            x = workspace.raw_ws.extractX()
        y = workspace.raw_ws.extractY()
        e = workspace.raw_ws.extractE()
    mdict = {'x': x, 'y': y, 'e': e}
    savemat(path, mdict=mdict)


def _save_cut_to_ascii(workspace, ws_name, output_path):
    # get integration ranges from the name
    int_ranges = ws_name[ws_name.find('('):]
    int_start = int_ranges[1:int_ranges.find(',')]
    int_end = int_ranges[int_ranges.find(',')+1:-1]
    ws_name = ws_name[:ws_name.find('_cut')]

    dim = workspace.raw_ws.getDimension(0)
    units = dim.getUnits()

    x, y, e = _get_md_histo_xye(workspace.raw_ws)
    header = 'MSlice Cut of workspace "%s" along "%s" between %s and %s' % (ws_name, units, int_start, int_end)
    out_data = np.c_[x, y, e]
    _output_data_to_ascii(output_path, out_data, header)


def _save_slice_to_ascii(workspace, output_path):
    header = 'MSlice Slice of workspace "%s"' % (workspace.name)
    x, y, e = _get_slice_mdhisto_xye(workspace.raw_ws)
    dim_sz = [workspace.raw_ws.getDimension(i).getNBins() for i in range(workspace.raw_ws.getNumDims())]
    nbins = np.prod(dim_sz)
    x = [np.reshape(x0, nbins) for x0 in np.broadcast_arrays(*x)]
    y = np.reshape(y, nbins)
    e = np.reshape(e, nbins)
    out_data = np.column_stack(tuple(x+[y, e]))
    _output_data_to_ascii(output_path, out_data, header)


def load_from_ascii(file_path, ws_name):
    file = open(file_path, 'r')
    header = file.readline()
    if not header.startswith("# MSlice Cut"):
        raise ValueError
    x, y, e = np.loadtxt(file).transpose()
    extents = str(np.min(x)) + ',' + str(np.max(x))
    nbins = len(x)
    units = header[header.find('along "'):header.find('" between')]
    run_algorithm('CreateMDHistoWorkspace', output_name=ws_name, SignalInput=y, ErrorInput=e, Dimensionality=1,
                  Extents=extents, NumberOfBins=nbins, Names='Dim1', Units=units)


def _get_md_histo_xye(histo_ws):
    dim = histo_ws.getDimension(0)
    start = dim.getMinimum()
    end = dim.getMaximum()
    nbin = dim.getNBins()
    x = np.linspace(start, end, nbin)
    y = histo_ws.getSignalArray()
    e = np.sqrt(histo_ws.getErrorSquaredArray())
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
    np.savetxt(str(output_path), out_data, fmt='%12.9e', header=header)
