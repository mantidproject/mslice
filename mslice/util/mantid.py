from __future__ import (absolute_import, division, print_function)
from six import iteritems
import mantid.simpleapi as s_api
from mantid.api import AlgorithmFactory, AnalysisDataService
from mslice.models.workspacemanager.workspace_provider import add_workspace
from mslice.models.projection.powder.make_projection import MakeProjection
from mslice.workspace import wrap_workspace
from mslice.workspace.base import WorkspaceBase as Workspace


def initialize_mantid():
    AlgorithmFactory.subscribe(MakeProjection)
    s_api._create_algorithm_function('MakeProjection', 1, MakeProjection())


def run_algorithm(alg_name, output_name=None, store=True, **kwargs):
    if isinstance(kwargs.get('InputWorkspace'), Workspace):
        kwargs['InputWorkspace'] = kwargs['InputWorkspace'].raw_ws
    if output_name is not None:
        kwargs['OutputWorkspace'] = output_name

    ws = getattr(s_api, alg_name)(StoreInADS=False, **kwargs)

    if store:
        ws = wrap_workspace(ws, output_name)
        add_workspace(ws, output_name)
    return ws

def run_ADS_dependent_algorithm(alg_name, output_name, input_workspace_dict):
    '''Need to wrap some algorithm calls because they don't like input workspaces that aren't in the ADS...'''
    for workspace in iteritems(input_workspace_dict):
        AnalysisDataService.Instance().addOrReplace(*workspace)
    ws = run_algorithm(alg_name, output_name=output_name, InputWorkspaces=input_workspace_dict.keys())
    for workspace in input_workspace_dict.keys():
        AnalysisDataService.Instance().remove(str(workspace))
    return ws
