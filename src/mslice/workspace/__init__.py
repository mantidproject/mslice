from mantid.api import IMDEventWorkspace, IMDHistoWorkspace
from .workspace import Workspace as MatrixWorkspace
from .pixel_workspace import PixelWorkspace
from .histogram_workspace import HistogramWorkspace

def wrap_workspace(raw_ws, name):
    if isinstance(raw_ws, IMDEventWorkspace):
        wrapped = PixelWorkspace(raw_ws, name)
    elif isinstance(raw_ws, IMDHistoWorkspace):
        wrapped = HistogramWorkspace(raw_ws, name)
    else:
        wrapped = MatrixWorkspace(raw_ws, name)
    return wrapped
