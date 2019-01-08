from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets

PACKAGES = {'mslice.cli': 'mc', 'matplotlib.pyplot': 'plt'}


def generate_script(figure, ws_name, filename=None, plot_handler=None):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(figure, 'Save File')
        if not path:
            return
        filename = path + '{}'.format('.py' if '.py' != path[-3:] else '')

    ws = get_workspace_handle(ws_name).raw_ws
    GeneratePythonScript(ws, Filename=filename)
    add_import_statements(filename)
    add_plot_statements(filename, plot_handler)


def add_import_statements(filename):
    with open(filename, 'r+') as generated_script:
        script_lines = generated_script.readlines()
        for i, statement in enumerate(import_statements()):
            script_lines.insert(3 + i, statement)  # Used 3 since the first 3 lines of generated scripts are comments
        generated_script.seek(0)
        generated_script.writelines(script_lines)


def add_plot_statements(filename, plot_handler):
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    from mslice.plotting.plot_window.cut_plot import CutPlot

    with open(filename, 'r+') as generated_script:
        script_lines = generated_script.readlines()
        line_no = len(script_lines)
        script_lines.insert(line_no - 1, '\n')
        script_lines[line_no] = 'ws = {}'.format(script_lines[line_no])

        if plot_handler is not None:
            if isinstance(plot_handler, SlicePlot):
                script_lines.append('slice_ws = mc.Slice(ws)')
                script_lines.append('ax.pcolormesh(slice_ws)')
            elif isinstance(plot_handler, CutPlot):
                script_lines.append('cut_ws = mc.Cut(ws)')
                script_lines.append('ax.plot(cut_ws)')

        generated_script.seek(0)
        generated_script.writelines(script_lines)


def extract_plot_options(plot_handler):
    from mslice.plotting.plot_window.slice_plot import SlicePlot
    from mslice.plotting.plot_window.cut_plot import CutPlot

    if isinstance(plot_handler, SlicePlot):
        return plot_handler.slice_plot_options
    elif isinstance(plot_handler, CutPlot):
        return plot_handler.cut_plot_options


def import_statements():
    statements = []
    for package in PACKAGES:
        statements.append('import {} as {}\n'.format(package, PACKAGES[package]))
    statements.append("fig, ax = plt.subplots(subplot_kw={'projection': 'mslice'})\n\n")
    return statements

