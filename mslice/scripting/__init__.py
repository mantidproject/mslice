import ctypes
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle, get_visible_workspace_names
from mslice.util.qt import QtWidgets
from mslice.scripting.helperfunctions import add_plot_statements
from mslice.app.presenters import get_cut_plotter_presenter
from mslice.workspace.base import WorkspaceBase


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

def _get_c_address(mslice_workspace):
    # Hack to get the c-address of the shared pointer to the workspace object (assumes a boost.python.class and CPython)
    mslice_workspace = get_workspace_handle(mslice_workspace)
    if isinstance(mslice_workspace, WorkspaceBase):
        return ctypes.cast(id(mslice_workspace.raw_ws), ctypes.POINTER(ctypes.c_void_p))[8]
    return None

def _parse_temp_ws_names(kwargs, current_ws_refs):
    idx = kwargs.find('__TMP') + 5
    ws_map = {}
    while idx > 0:
        addr_str = kwargs[idx:kwargs.find("'", idx)]
        ws_map[int(addr_str, 16)] = addr_str
        idx_tmp = kwargs.find('__TMP', idx)
        idx =  idx + idx_tmp + 5 if idx_tmp > 0 else idx_tmp
    for addr, addr_str in list(ws_map.items()):
        if addr in current_ws_refs.keys():
            kwargs = kwargs.replace("'__TMP{}'".format(addr_str), current_ws_refs[addr])
    return kwargs

def generate_script_lines(raw_ws, workspace_name):
    lines = []
    ws_name = workspace_name.replace(".", "_")
    alg_history = raw_ws.getHistory().getAlgorithmHistories()
    existing_ws_refs = []
    current_ws_refs = {_get_c_address(ws): ws.replace(".", "_") for ws in get_visible_workspace_names()}
    for algorithm in alg_history:
        alg_name = algorithm.name()
        kwargs, output_ws = get_algorithm_kwargs(algorithm, existing_ws_refs)
        if '__TMP' in kwargs:
            kwargs = _parse_temp_ws_names(kwargs, current_ws_refs)
        if output_ws is not None and output_ws != ws_name:
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
            if prop.name() == "OutputWorkspace":
                output_ws = prop.value().replace(".", "_")
            if algorithm.name() == "Load":
                if prop.name() == "Filename":
                    arguments += ["{}='{}'".format(prop.name(), prop.value())]
                    continue
                elif prop.name() == "LoaderName" or prop.name() == "LoaderVersion":
                    continue
            elif algorithm.name() == "MakeProjection":
                if prop.name() == "Limits" or prop.name() == "OutputWorkspace" or prop.name() == "ProjectionType":
                    continue
            if isinstance(prop.value(), str) and prop.value().replace(".", "_") in existing_ws_refs:
                arguments += ["{}={}".format(prop.name(), prop.value().replace(".", "_"))]
            elif isinstance(prop.value(), str):
                arguments += ["{}='{}'".format(prop.name(), prop.value())]
            else:
                arguments += ["{}={}".format(prop.name(), prop.value())]
    return ", ".join(arguments), output_ws
