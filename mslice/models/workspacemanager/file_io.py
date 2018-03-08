from __future__ import (absolute_import, division, print_function)
import os.path
from mantid.api import IMDEventWorkspace, IMDHistoWorkspace
from mantid.simpleapi import CreateMDHistoWorkspace, SaveMD, SaveNexus, SaveAscii
from mslice.util.qt.QtWidgets import QFileDialog

import numpy as np
from scipy.io import savemat


def get_save_directory(multiple_files, save_as_image=False, default_ext=None):
    '''
    Show file dialog so user can choose where to save.
    :param multiple_files: boolean - whether more than one file is being saved
    :param default_ext: file extension that is selected by default
    :return: path to save directory, name to save the file as, file format extension
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
            ext_to_qtfilter = {'.nxs': 'Nexus (*.nxs)', '.txt': 'Ascii (*.txt)', '.mat': 'Matlab (*.mat)'}
            file_dialog.selectNameFilter(ext_to_qtfilter[default_ext])
        if (file_dialog.exec_()):
            path = file_dialog.selectedFiles()[0] + file_dialog.selectedFilter()[-5:-1]
            ext = path[path.rfind('.'):]
            return os.path.dirname(path), os.path.basename(path), ext
        else:
            raise RuntimeError("dialog cancelled")


def save_nexus(workspace, path):
    if isinstance(workspace, IMDEventWorkspace) or isinstance(workspace, IMDHistoWorkspace):
        SaveMD(InputWorkspace=workspace.name(), Filename=path)
    else:
        SaveNexus(InputWorkspace=workspace.name(), Filename=path)


def save_ascii(workspace, path):
    if isinstance(workspace, IMDEventWorkspace):
        raise RuntimeError("Cannot save MDEventWorkspace as ascii")
    elif isinstance(workspace, IMDHistoWorkspace):
        _save_cut_to_ascii(workspace, workspace.name(), path)
    else:
        SaveAscii(InputWorkspace=workspace, Filename=path)


def save_matlab(workspace, path):
    if isinstance(workspace, IMDEventWorkspace):
        raise RuntimeError("Cannot save MDEventWorkspace as Matlab file")
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