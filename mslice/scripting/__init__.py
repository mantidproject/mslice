from mantid.simpleapi import GeneratePythonScript
from mslice.models.workspacemanager.workspace_provider import get_workspace_handle


def generate_script(ws_name):
    ws = get_workspace_handle(ws_name).raw_ws
    GeneratePythonScript(ws, Filename="test.py")