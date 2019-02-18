from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets
from mslice.scripting.helperfunctions import add_plot_statements, cleanup
from mslice.app.presenters import get_cut_plotter_presenter


def generate_script(ws_name, filename=None, plot_handler=None, window=None):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(window, 'Save File')
        if not path:
            return
        filename = path + '{}'.format('' if path.endswith('.py') else '.py')
    else:
        filename = filename + '{}'.format('' if filename.endswith('.py') else '.py')

    script_lines = preprocess_lines(ws_name, plot_handler)
    script_lines = add_plot_statements(script_lines, plot_handler)
    with open(filename, 'w') as generated_script:
        generated_script.writelines(script_lines)


def preprocess_lines(ws_name, plot_handler):
    from mslice.plotting.plot_window.cut_plot import CutPlot

    if isinstance(plot_handler, CutPlot) and len(get_cut_plotter_presenter()._cut_cache) > 1:
        script_lines = []
        for workspace_name in get_cut_plotter_presenter()._cut_cache:
            ws = get_workspace_handle(workspace_name).raw_ws
            script_lines += ["\nws_{} = mc.".format(workspace_name.replace(".", "_"))] + cleanup(
                GeneratePythonScript(ws).split('\n'))
    else:
        ws = get_workspace_handle(ws_name).raw_ws
        script_lines = ["\nws = mc."] + cleanup(GeneratePythonScript(ws).split('\n'))

    return script_lines
