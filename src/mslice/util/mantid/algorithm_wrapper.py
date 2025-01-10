from uuid import uuid4

from mslice.models.workspacemanager.workspace_provider import add_workspace, get_workspace_handle
from mantid.api import AnalysisDataService, AlgorithmManager, Workspace

from mslice.workspace import wrap_workspace
from mslice.workspace.base import WorkspaceBase as MsliceWorkspace
from mslice.workspace.workspace import Workspace as MsliceWorkspace2D


def _parse_ws_names(args, kwargs):
    input_workspace = kwargs.get('InputWorkspace', None)
    if input_workspace:
        kwargs['InputWorkspace'] = _name_or_wrapper_to_workspace(input_workspace)
    elif len(args) > 0:
        if isinstance(args[0], MsliceWorkspace) or isinstance(args[0], str):
            input_workspace = get_workspace_handle(args[0])
        args = (_name_or_wrapper_to_workspace(args[0]),) + args[1:]

    output_name = ''
    if 'OutputWorkspace' in kwargs:
        output_name = kwargs.pop('OutputWorkspace')

    for key in kwargs.keys():
        if input_workspace is None and 'LHS' in key:
            input_workspace = get_workspace_handle(kwargs[key])
        if 'Input' not in key and 'Output' not in key:
            if isinstance(kwargs[key], MsliceWorkspace):
                kwargs[key] = _name_or_wrapper_to_workspace(kwargs[key])

    return input_workspace, output_name, args, kwargs


def _alg_has_outputws(wrapped_alg):
    alg = AlgorithmManager.create(wrapped_alg.__name__)
    return any(['OutputWorkspace' in prop.name for prop in alg.getProperties()])


def wrap_algorithm(algorithm):
    def alg_wrapper(*args, **kwargs):
        input_workspace, output_name, args, kwargs = _parse_ws_names(args, kwargs)

        args = tuple([_name_or_wrapper_to_workspace(arg) if isinstance(arg, MsliceWorkspace) else arg for arg in args])

        if 'InputWorkspaces' in kwargs:
            kwargs['InputWorkspaces'] = [_name_or_wrapper_to_workspace(arg) for arg in kwargs['InputWorkspaces']]

        for ky in [k for k in kwargs.keys() if 'Workspace' in k]:
            if isinstance(kwargs[ky], str) and '__MSL' not in kwargs[ky]:
                kwargs[ky] = _name_or_wrapper_to_workspace(kwargs[ky])

        if _alg_has_outputws(algorithm):
            ads_name = '__MSL' + output_name if output_name else '__MSLTMP' + str(uuid4())[:8]
            store = kwargs.pop('store', True)
            if not store:
                ads_name += '_HIDDEN'
            result = algorithm(*args, OutputWorkspace=ads_name, **kwargs)
        else:
            result = algorithm(*args, **kwargs)

        if isinstance(result, Workspace):
            if isinstance(input_workspace, MsliceWorkspace2D) and isinstance(result, type(input_workspace.raw_ws)):
                result = get_workspace_handle(input_workspace).rewrap(result)
                result.name = output_name
            else:
                result = wrap_workspace(result, output_name)
            if store:
                add_workspace(result, output_name)
                from mslice.app import is_gui
                if is_gui():
                    from mslice.app.presenters import get_slice_plotter_presenter
                    get_slice_plotter_presenter().update_displayed_workspaces()
        return result

    return alg_wrapper


def _name_or_wrapper_to_workspace(input_ws):
    if isinstance(input_ws, MsliceWorkspace):
        return input_ws.raw_ws
    elif isinstance(input_ws, str):
        return get_workspace_handle(input_ws).raw_ws
    else:
        return input_ws


def add_to_ads(workspaces):
    try:
        workspaces = iter(workspaces)
    except TypeError:
        workspaces = [workspaces]
    for workspace in workspaces:
        startid = (5 if workspace.name.startswith('__mat') else 2) if workspace.name.startswith('__') else 0
        AnalysisDataService.Instance().addOrReplace(workspace.name[startid:], workspace.raw_ws)


def remove_from_ads(workspacename):
    if AnalysisDataService.Instance().doesExist(workspacename):
        AnalysisDataService.Instance().remove(workspacename)
    # Remove hidden workspaces from ADS
    hiddenworkspace = '__MSL' + workspacename
    if AnalysisDataService.Instance().doesExist(hiddenworkspace):
        AnalysisDataService.Instance().remove(hiddenworkspace)
    hiddenworkspace = '__MSL__mat' + workspacename
    if AnalysisDataService.Instance().doesExist(hiddenworkspace):
        AnalysisDataService.Instance().remove(hiddenworkspace)
    hiddenworkspace = '__MSL' + workspacename + '_HIDDEN'
    print(AnalysisDataService.Instance().getObjectNames())
    if AnalysisDataService.Instance().doesExist(hiddenworkspace):
        AnalysisDataService.Instance().remove(hiddenworkspace)
