from mslice.workspace.base import WorkspaceBase as Workspace
from mslice.workspace.histogram_workspace import HistogramWorkspace


# Arguments Validation
def validate_args(*args):
    """
    Checks if args[0] is a WorkspaceBase or HistogramWorkspace
    :param args: arguments passed into a function
    :return:  Returns True if args[0] is an instance of WorkspaceBase or HistogramWorkspace
    """
    return len(args) > 0 and (isinstance(args[0], Workspace)
                              or (isinstance(args[0], HistogramWorkspace)))


def is_cut(*args):
    """
    Checks if arguments are suitable for
    :param args:
    :return:
    """
    return len(args) > 0 and (isinstance(args[0], HistogramWorkspace))
