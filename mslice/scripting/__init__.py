from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets

PACKAGES = {'mslice.cli': 'mc', 'matplotlib.pyplot': 'plt'}


def generate_script(figure, ws_name, filename=None):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(figure, 'Save File')
        if not path:
            return
        filename = path + '{}'.format('.py' if '.py' not in path[-3:] else '')

    ws = get_workspace_handle(ws_name).raw_ws
    GeneratePythonScript(ws, Filename=filename)
    add_import_statements(filename)


def add_import_statements(filename):
    with open(filename, 'r+') as generated_script:
        script_lines = generated_script.readlines()
        for i, statement in enumerate(import_statements()):
            script_lines.insert(3 + i, statement)
        generated_script.seek(0)
        generated_script.writelines(script_lines)


def import_statements():
    statements = []
    for package in PACKAGES:
        statements.append('import {} as {} \n'.format(package, PACKAGES[package]))
    return statements

