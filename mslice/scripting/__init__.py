from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets


def generate_script(plot_window, ws_name, filename):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(plot_window, 'Save File', '', 'Python (.py)')
        if path is u'':
            raise RuntimeError("save_file_dialog cancelled")
        filename = path
    ws = get_workspace_handle(ws_name).raw_ws
    GeneratePythonScript(ws, Filename=filename)
