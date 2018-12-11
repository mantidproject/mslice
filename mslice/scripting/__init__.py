from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets


def generate_script(figure, ws_name, filename=None):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(figure, 'Save File')
        if not path:
            return
        filename = path
    ws = get_workspace_handle(ws_name).raw_ws
    GeneratePythonScript(ws, Filename=filename + '.py')
