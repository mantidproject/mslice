from qtpy import QtGui, QtWidgets
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle
from mslice.scripting.helperfunctions import (
    add_plot_statements,
    replace_ws_special_chars,
)
from mslice.app.presenters import get_cut_plotter_presenter


def generate_script(
    ws_name, filename=None, plot_handler=None, window=None, clipboard=False
):
    if filename is None and not clipboard:
        path = QtWidgets.QFileDialog.getSaveFileName(window, "Save File")
        if isinstance(path, tuple):
            path = path[0]
        if not path:
            return
        filename = f"{path}{'' if path.endswith('.py') else '.py'}"
    elif not clipboard:
        filename = f"{filename}{'' if filename.endswith('.py') else '.py'}"

    ax = window.canvas.figure.axes[0]
    script_lines = preprocess_lines(ws_name, plot_handler, ax)
    script_lines = add_plot_statements(script_lines, plot_handler, ax)
    if clipboard:
        cb = QtGui.QGuiApplication.clipboard()
        cb.setText("".join(script_lines), mode=cb.Clipboard)
    else:
        with open(filename, "w") as generated_script:
            generated_script.writelines(script_lines)


def preprocess_lines(ws_name, plot_handler, ax):
    from mslice.plotting.plot_window.cut_plot import CutPlot

    script_lines = []
    if isinstance(plot_handler, CutPlot):
        cache_list = get_cut_plotter_presenter()._cut_cache_dict[ax]
        ws_list = {}  # use a dict to ensure unique workspaces
        for workspace_name in [cut.parent_ws_name for cut in cache_list]:
            ws_list[get_workspace_handle(workspace_name).raw_ws] = workspace_name
        for ws, workspace_name in list(ws_list.items()):
            script_lines += generate_script_lines(ws, workspace_name)
    else:
        ws = get_workspace_handle(ws_name).raw_ws
        script_lines += generate_script_lines(ws, ws_name)

    return script_lines


def generate_script_lines(raw_ws, workspace_name):
    lines = []
    ws_name = replace_ws_special_chars(workspace_name)
    alg_history = raw_ws.getHistory().getAlgorithmHistories()
    existing_ws_refs = []

    # Loop back from end to see if we have a Save / Load pair and truncate at the load if so
    prev_algo = alg_history[-1]
    for idx, algorithm in reversed(list(enumerate(alg_history[:-1]))):
        new_algo = algorithm
        if "Save" in new_algo.name() and "Load" in prev_algo.name():
            alg_history = alg_history[idx + 1 :]
            break
        else:
            prev_algo = new_algo

    for algorithm in alg_history:
        alg_name = algorithm.name()
        kwargs, output_ws = get_algorithm_kwargs(algorithm, existing_ws_refs)
        if output_ws is not None and output_ws != ws_name:
            lines += [f"{output_ws} = mc.{alg_name}({kwargs})\n"]
            existing_ws_refs.append(output_ws)
        else:
            lines += [f"ws_{ws_name} = mc.{alg_name}({kwargs})\n"]
    return lines


def _parse_prop(prop):
    pval = prop.value()
    pname = prop.name()
    hidden = False
    output_ws = None
    if isinstance(pval, str):
        pval = pval.replace("__MSL", "").replace("_HIDDEN", "")
    if prop.name() == "OutputWorkspace":
        output_ws = replace_ws_special_chars(pval)
        if "_HIDDEN" in prop.value():
            hidden = True
    return pname, pval, output_ws, hidden


def get_algorithm_kwargs(algorithm, existing_ws_refs):
    arguments = []
    output_ws = None
    for prop in algorithm.getProperties():
        if not prop.isDefault():
            pname, pval, tmp_output_ws, hidden = _parse_prop(prop)
            if pname == "OutputWorkspace":
                output_ws = tmp_output_ws
            if hidden:
                arguments += ["store=False"]
                continue
            if algorithm.name() == "Load":
                if prop.name() == "Filename":
                    arguments += [f"{prop.name()}=r'{pval}'"]
                    continue
                elif prop.name() == "LoaderName" or prop.name() == "LoaderVersion":
                    continue
            elif algorithm.name() == "MakeProjection":
                if (
                    prop.name() == "Limits"
                    or prop.name() == "OutputWorkspace"
                    or prop.name() == "ProjectionType"
                ):
                    continue
            if (
                isinstance(pval, str)
                and replace_ws_special_chars(pval) in existing_ws_refs
            ):
                arguments += [f"{prop.name()}={replace_ws_special_chars(pval)}"]
            elif isinstance(prop.value(), str):
                arguments += [f"{prop.name()}='{pval}'"]
            else:
                arguments += [f"{prop.name()}={pval}"]
    return ", ".join(arguments), output_ws
