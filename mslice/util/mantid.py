from __future__ import (absolute_import, division, print_function)

import mantid.simpleapi as s_api
from mantid.api import AlgorithmFactory
from mslice.models.workspacemanager.mantid_workspace_provider import _loaded_workspaces
from mslice.models.projection.powder.make_projection import MakeProjection
from mslice.workspace import wrap_workspace
from mslice.workspace.base import WorkspaceBase as Workspace


def initialize_mantid():
    AlgorithmFactory.subscribe(MakeProjection)
    s_api._create_algorithm_function('MakeProjection', 1, MakeProjection)


def run_alg(alg_name, output_name=None, store=True, **kwargs):
    return run_algorithm(getattr(s_api, alg_name), output_name, store, **kwargs)

def run_algorithm(algorithm, output_name=None, store=True, **kwargs):
    if isinstance(kwargs.get('InputWorkspace'), Workspace):
        kwargs['InputWorkspace'] = kwargs['InputWorkspace'].raw_ws
    if output_name is not None:
        kwargs['OutputWorkspace'] = output_name

    ws = algorithm(**kwargs)

    if store:
        ws = wrap_workspace(ws, output_name)
        _add_workspace(ws, output_name)
    return ws

