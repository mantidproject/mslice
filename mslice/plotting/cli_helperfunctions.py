from mslice.workspace.base import WorkspaceBase as Workspace
from mslice.workspace.histogram_workspace import HistogramWorkspace


# Arguments Validation
def validate_args(*args):
    return len(args) > 0 and (isinstance(args[0], Workspace)
                              or (isinstance(args[0], HistogramWorkspace)))
