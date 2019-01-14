from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets
from mslice.scripting.helperfunctions import add_plot_statements, add_import_statements


def generate_script(figure, ws_name, filename=None, plot_handler=None):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(figure, 'Save File')
        if not path:
            return
        filename = path + '{}'.format('.py' if '.py' not in path[-3:] else '')

    ws = get_workspace_handle(ws_name).raw_ws
    GeneratePythonScript(ws, Filename=filename)
    add_import_statements(filename)
    add_plot_statements(filename, plot_handler)
