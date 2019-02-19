from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.util.qt import QtWidgets
from mslice.scripting.helperfunctions import add_plot_statements
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
    script_lines = []

    if isinstance(plot_handler, CutPlot) and len(get_cut_plotter_presenter()._cut_cache) > 1:
        cut_cache = get_cut_plotter_presenter()._cut_cache
        cache_list = get_cut_plotter_presenter()._cut_cache_list
        for workspace_name in cut_cache:
            if any([workspace_name.replace(".", "_") == cut.workspace_name for cut in cache_list]):
                ws = get_workspace_handle(workspace_name).raw_ws
                script_lines += generate_script_lines(ws, workspace_name)
    else:
        ws = get_workspace_handle(ws_name).raw_ws
        script_lines += generate_script_lines(ws, ws_name)

    return script_lines


def generate_script_lines(raw_ws, workspace_name):
    lines = []
    ws_name = workspace_name.replace(".", "_")
    alg_history = raw_ws.getHistory().getAlgorithmHistories()[::-1]
    for algorithm in alg_history:
        alg_name = algorithm.name()
        kwargs = get_algorithm_kwargs(algorithm, ws_name)
        if alg_name != 'Load':
            lines += ["ws_{} = mc.{}({})\n".format(ws_name, alg_name, kwargs)]
        else:
            lines += ["ws_{} = mc.{}({})\n".format(ws_name, alg_name, kwargs)]
            break
    return lines[::-1]


def get_algorithm_kwargs(algorithm, workspace_name):
    arguments = []
    if algorithm.name() == "Load":
        for property in algorithm.getProperties():
            if property.name() == "Filename":
                arguments = ["{}='{}'".format(property.name(), property.value())]
    elif algorithm.name() == "MakeProjection":
        for property in algorithm.getProperties():
            if not property.isDefault():
                if property.name() == "InputWorkspace":
                    arguments += ["{}=ws_{}".format(property.name(), workspace_name)]
                elif property.name() == "OutputWorkspace" or property.name() == "Limits":
                    pass
                else:
                    arguments += ["{}='{}'".format(property.name(), property.value())]
    else:
        for property in algorithm.getProperties():
            if not property.isDefault():
                if property.name() == "Filename":
                    arguments = ["{}='{}'".format(property.name(), property.value())]
                else:
                    arguments += ["{}=ws_{}".format(property.name(), property.value())]
    return ", ".join(arguments)
