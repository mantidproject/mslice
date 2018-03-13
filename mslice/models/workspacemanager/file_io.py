from __future__ import (absolute_import, division, print_function)
import os.path
from mantid.api import IMDHistoWorkspace
from mantid.simpleapi import CreateMDHistoWorkspace, SaveMD, SaveNexus, SaveAscii
from mslice.util.qt.QtWidgets import QFileDialog

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
            path = file_dialog.selectedFiles()[0]
            if not path.rfind("."): # add extension unless there's one in the name
                path += file_dialog.selectedFilter()[-5:-1]
            ext = path[path.rfind('.'):]
            return os.path.dirname(path), os.path.basename(path), ext
        else:
            raise RuntimeError("dialog cancelled")


def save_nexus(workspace, path, is_slice):
    if is_slice:
        SaveMD(InputWorkspace=workspace.name()[2:], Filename=path)
    elif isinstance(workspace, IMDHistoWorkspace):
        SaveMD(InputWorkspace=workspace.name(), Filename=path)
    else:
        SaveNexus(InputWorkspace=workspace.name(), Filename=path)


def save_ascii(workspace, path, is_slice):
    if is_slice:
        _save_slice_to_ascii(workspace, path)
    elif isinstance(workspace, IMDHistoWorkspace):
        _save_cut_to_ascii(workspace, workspace.name(), path)
    else:
        SaveAscii(InputWorkspace=workspace, Filename=path)


def save_matlab(workspace, path, is_slice):
    if is_slice:
        x, y, e = _get_slice_mdhisto_xye(workspace)
    elif isinstance(workspace, IMDHistoWorkspace):
        x, y, e = _get_md_histo_xye(workspace)
    else:
        x = workspace.extractX()
        y = workspace.extractY()
        e = workspace.extractE()
    mdict = {'x': x, 'y': y, 'e': e}
    savemat(path, mdict=mdict)


def _save_cut_to_ascii(workspace, ws_name, output_path):
    # get integration ranges from the name
    int_ranges = ws_name[ws_name.find('('):]
    int_start = int_ranges[1:int_ranges.find(',')]
    int_end = int_ranges[int_ranges.find(',')+1:-1]
    ws_name = ws_name[:ws_name.find('_cut')]

    dim = workspace.getDimension(0)
    units = dim.getUnits()

    x, y, e = _get_md_histo_xye(workspace)
    header = 'MSlice Cut of workspace "%s" along "%s" between %s and %s' % (ws_name, units, int_start, int_end)
    out_data = np.c_[x, y, e]
    np.savetxt(str(output_path), out_data, fmt='%12.9e', header=header)


def _save_slice_to_ascii(workspace, output_path):
    header = 'MSlice Slice of workspace "%s"' % (workspace.name())
    x, y, e = _get_slice_mdhisto_xye(workspace)
    dim_sz = [workspace.getDimension(i).getNBins() for i in range(workspace.getNumDims())]
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
    CreateMDHistoWorkspace(SignalInput=y, ErrorInput=e, Dimensionality=1, Extents=extents, NumberOfBins=nbins,
                           Names='Dim1', Units=units, OutputWorkspace=ws_name)


def _get_md_histo_xye(histo_ws):
    dim = histo_ws.getDimension(0)
    start = dim.getMinimum()
    end = dim.getMaximum()
    step = dim.getBinWidth()
    x = np.arange(start, end, step)
    y = histo_ws.getSignalArray()
    e = np.sqrt(histo_ws.getErrorSquaredArray())
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
    return x, y, e


def _output_data_to_ascii(output_path, out_data, header):
    np.savetxt(str(output_path), out_data, fmt='%12.9e', header=header)
