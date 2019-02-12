from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets
from mslice.scripting.helperfunctions import add_plot_statements, cleanup


def generate_script(ws_name, filename=None, window=None, plot_handler=None):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(window, 'Save File')
        if not path:
            return
        filename = path + '{}'.format('' if path.endswith('.py') else '.py')
    else:
        filename = filename + '{}'.format('' if filename.endswith('.py') else '.py')

    ws = get_workspace_handle(ws_name).raw_ws
    script_lines = cleanup(GeneratePythonScript(ws).split('\n'))
    script_lines = add_plot_statements(script_lines, plot_handler)

    with open(filename, 'w') as generated_script:
        generated_script.writelines(script_lines)
