from __future__ import (absolute_import, division, print_function)

from uuid import uuid4
from six import string_types

from mslice.models.workspacemanager.workspace_provider import add_workspace, get_workspace_handle
from mantid.api import AnalysisDataService, AlgorithmManager, Workspace

from mslice.workspace import wrap_workspace
from mslice.workspace.base import WorkspaceBase as MsliceWorkspace
from mslice.workspace.workspace import Workspace as MsliceWorkspace2D

def _parse_ws_names(args, kwargs):
    input_workspace = None
    if 'InputWorkspace' in kwargs:
        input_workspace = kwargs['InputWorkspace']
        kwargs['InputWorkspace'] = _name_or_wrapper_to_workspace(kwargs['InputWorkspace'])
    elif len(args) > 0:
        input_workspace = args[0]
        args = (_name_or_wrapper_to_workspace(args[0]),) + args[1:]

    output_name = ''
    if 'OutputWorkspace' in kwargs:
        output_name = kwargs.pop('OutputWorkspace')

    for ws in [k for k in kwargs.keys() if isinstance(kwargs[k], MsliceWorkspace)]:
        if input_workspace is None and 'LHS' in ws:
            input_workspace = kwargs[ws]
        if 'Input' not in ws and 'Output' not in ws:
            kwargs[ws] = _name_or_wrapper_to_workspace(kwargs[ws])

    return (input_workspace, output_name, args, kwargs)

def _alg_has_outputws(wrapped_alg):
    alg = AlgorithmManager.create(wrapped_alg.func_name)
    return any(['OutputWorkspace' in prop.name for prop in alg.getProperties()])

def wrap_algorithm(algorithm):
    def alg_wrapper(*args, **kwargs):

        input_workspace, output_name, args, kwargs = _parse_ws_names(args, kwargs)

        args = tuple([_name_or_wrapper_to_workspace(arg) if isinstance(arg, MsliceWorkspace) else arg for arg in args])

        if 'InputWorkspaces' in kwargs:
            kwargs['InputWorkspaces'] = [_name_or_wrapper_to_workspace(arg) for arg in kwargs['InputWorkspaces']]

        for ky in [k for k in kwargs.keys() if 'Workspace' in k]:
            if isinstance(kwargs[ky], string_types) and '__MSL' not in kwargs[ky]:
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
    elif isinstance(input_ws, string_types):
        return get_workspace_handle(input_ws).raw_ws
    else:
        return input_ws


def add_to_ads(workspaces):
    try:
        workspaces = iter(workspaces)
    except TypeError:
        workspaces = [workspaces]
    for workspace in workspaces:
        AnalysisDataService.Instance().addOrReplace(workspace.name, workspace.raw_ws)
