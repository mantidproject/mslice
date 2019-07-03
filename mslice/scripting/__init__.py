from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, get_visible_workspace_names
from mslice.util.qt import QtWidgets
from mslice.scripting.helperfunctions import add_plot_statements
from mslice.app.presenters import get_cut_plotter_presenter
from six import string_types


def generate_script(ws_name, filename=None, plot_handler=None, window=None):
    if filename is None:
        path = QtWidgets.QFileDialog.getSaveFileName(window, 'Save File')
        if isinstance(path, tuple):
            path = path[0]
        if not path:
            return
        filename = path + '{}'.format('' if path.endswith('.py') else '.py')
    else:
        filename = filename + '{}'.format('' if filename.endswith('.py') else '.py')

    ax = window.canvas.figure.axes[0]
    script_lines = preprocess_lines(ws_name, plot_handler, ax)
    script_lines = add_plot_statements(script_lines, plot_handler, ax)
    with open(filename, 'w') as generated_script:
        generated_script.writelines(script_lines)


def preprocess_lines(ws_name, plot_handler, ax):
    from mslice.plotting.plot_window.cut_plot import CutPlot
    script_lines = []
    if isinstance(plot_handler, CutPlot):
        cache_list = get_cut_plotter_presenter()._cut_cache_dict[ax]
        ws_list = {}    # use a dict to ensure unique workspaces
        for workspace_name in [cut.workspace_raw_name for cut in cache_list]:
            ws_list[get_workspace_handle(workspace_name).raw_ws] = workspace_name
        for ws, workspace_name in list(ws_list.items()):
            script_lines += generate_script_lines(ws, workspace_name)
    else:
        ws = get_workspace_handle(ws_name).raw_ws
        script_lines += generate_script_lines(ws, ws_name)

    return script_lines

def generate_script_lines(raw_ws, workspace_name):
    lines = []
    ws_name = workspace_name.replace(".", "_")
    alg_history = raw_ws.getHistory().getAlgorithmHistories()
    existing_ws_refs = []
    for algorithm in alg_history:
        alg_name = algorithm.name()
        kwargs, output_ws = get_algorithm_kwargs(algorithm, existing_ws_refs)
        if (output_ws is not None and output_ws != ws_name):
            lines += ["{} = mc.{}({})\n".format(output_ws, alg_name, kwargs)]
            existing_ws_refs.append(output_ws)
        else:
            lines += ["ws_{} = mc.{}({})\n".format(ws_name, alg_name, kwargs)]
    return lines


def get_algorithm_kwargs(algorithm, existing_ws_refs):
    arguments = []
    output_ws = None
    for prop in algorithm.getProperties():
        if not prop.isDefault():
            pval = prop.value()
            if isinstance(pval, string_types):
                pval = pval.replace("__MSL", "").replace("_HIDDEN", "")
            if prop.name() == "OutputWorkspace":
                output_ws = pval.replace(".", "_")
                if "_HIDDEN" in prop.value():
                    arguments += ["store=False"]
                    continue
            if algorithm.name() == "Load":
                if prop.name() == "Filename":
                    arguments += ["{}='{}'".format(prop.name(), pval)]
                    continue
                elif prop.name() == "LoaderName" or prop.name() == "LoaderVersion":
                    continue
            elif algorithm.name() == "MakeProjection":
                if prop.name() == "Limits" or prop.name() == "OutputWorkspace" or prop.name() == "ProjectionType":
                    continue
            if isinstance(pval, str) and pval.replace(".", "_") in existing_ws_refs:
                arguments += ["{}={}".format(prop.name(), pval.replace(".", "_"))]
            elif isinstance(prop.value(), str):
                arguments += ["{}='{}'".format(prop.name(), pval)]
            else:
                arguments += ["{}={}".format(prop.name(), pval)]
    return ", ".join(arguments), output_ws
