from __future__ import (absolute_import, division, print_function)
from contextlib import contextmanager
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
    print(alg_name)
    if isinstance(kwargs.get('InputWorkspace'), Workspace):
        kwargs['InputWorkspace'] = kwargs['InputWorkspace'].raw_ws
    if output_name is not None:
        kwargs['OutputWorkspace'] = output_name

    ws = getattr(s_api, alg_name)(StoreInADS=False, **kwargs)

    if store:
        ws = wrap_workspace(ws, output_name)
        add_workspace(ws, output_name)
    return ws


@contextmanager
def add_to_ads(workspaces):
    '''Need to wrap some algorithm calls because they don't like input workspaces that aren't in the ADS...'''
    for workspace in workspaces:
        AnalysisDataService.Instance().addOrReplace(workspace.name, workspace.raw_ws)
    yield
    for workspace in workspaces:
        AnalysisDataService.Instance().remove(str(workspace))