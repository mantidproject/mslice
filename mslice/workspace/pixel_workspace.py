from workspace import Workspace
from mantid.api import IMDEventWorkspace

class PixelWorkspace(Workspace):
    def __init__(self, pixel_workspace):
        pws = IMDEventWorkspace()
        # pws.