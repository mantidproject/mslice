from __future__ import (absolute_import, division, print_function)

from contextlib import contextmanager

import mantid.simpleapi as s_api
from mantid.api import AlgorithmFactory, AnalysisDataService
from mslice.models.workspacemanager.workspace_provider import add_workspace
from mslice.models.cut.cut import Cut
from mslice.models.projection.powder.make_projection import MakeProjection
from mslice.models.slice.slice import Slice
from mslice.workspace import wrap_workspace
from mslice.workspace.base import WorkspaceBase as Workspace


def initialize_mantid():
    AlgorithmFactory.subscribe(MakeProjection)
    AlgorithmFactory.subscribe(Slice)
    AlgorithmFactory.subscribe(Cut)
    s_api._create_algorithm_function('MakeProjection', 1, MakeProjection())
    s_api._create_algorithm_function('Slice', 1, Slice())
s_api._create_algorithm_function('Cut', 1, Cut())


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


@contextmanager
def add_to_ads(workspaces):
    '''Need to wrap some algorithm calls because they don't like input workspaces that aren't in the ADS...'''
    for workspace in workspaces:
        AnalysisDataService.Instance().addOrReplace(workspace.name, workspace.raw_ws)
    yield
    for workspace in workspaces:
        AnalysisDataService.Instance().remove(workspace.name)

