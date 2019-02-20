from __future__ import (absolute_import, division, print_function)

from contextlib import contextmanager

from mslice.models.workspacemanager.workspace_provider import add_workspace, get_workspace_handle
from mantid.api import AnalysisDataService, Workspace

from mslice.workspace import wrap_workspace
from mslice.workspace.base import WorkspaceBase as MsliceWorkspace


def wrap_algorithm(algorithm):
    def alg_wrapper(*args, **kwargs):
        output_name = ''
        if 'InputWorkspace' in kwargs:
            kwargs['InputWorkspace'] = _name_or_wrapper_to_workspace(kwargs['InputWorkspace'])
        elif len(args) > 0:
            args = (_name_or_wrapper_to_workspace(args[0]),) + args[1:]
        if 'OutputWorkspace' in kwargs:
            output_name = kwargs['OutputWorkspace']
        store = kwargs.pop('store', True)
        for ws in [k for k in kwargs.keys() if isinstance(kwargs[k], MsliceWorkspace)]:
            if 'Input' not in ws and 'Output' not in ws:
                kwargs[ws] = _name_or_wrapper_to_workspace(kwargs[ws])
        args = tuple([_name_or_wrapper_to_workspace(arg) if isinstance(arg, MsliceWorkspace) else arg for arg in args])

        result = algorithm(*args, StoreInADS=False, **kwargs)
        if isinstance(result, Workspace):
            result = wrap_workspace(result, output_name)
            if store:
                add_workspace(result, output_name)
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


@contextmanager
def wrap_in_ads(workspaces):
    """Need to wrap some algorithm calls because they don't like input workspaces that aren't in the ADS..."""
    for workspace in workspaces:
        AnalysisDataService.Instance().addOrReplace(workspace.name, workspace.raw_ws)
    yield
    for workspace in workspaces:
        AnalysisDataService.Instance().remove(workspace.name)


def add_to_ads(workspaces):
    try:
        workspaces = iter(workspaces)
    except TypeError:
        workspaces = [workspaces]
    for workspace in workspaces:
        AnalysisDataService.Instance().addOrReplace(workspace.name, workspace.raw_ws)
