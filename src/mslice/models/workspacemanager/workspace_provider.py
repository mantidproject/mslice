from mslice.workspace.base import WorkspaceBase as Workspace

_loaded_workspaces = {}


def get_workspace_handle(workspace_name):
    """"Return handle to workspace given workspace_name_as_string"""
    # if passed a workspace handle return the handle
    if isinstance(workspace_name, Workspace):
        return workspace_name
    try:
        return _loaded_workspaces[workspace_name]
    except KeyError:
        raise KeyError('workspace %s could not be found.' % workspace_name)


def add_workspace(workspace, name):
    _loaded_workspaces[name] = workspace


def remove_workspace(workspace):
    workspace = get_workspace_handle(workspace)
    del _loaded_workspaces[workspace.name]


def rename_workspace(workspace, new_name):
    workspace = get_workspace_handle(workspace)
    _loaded_workspaces[new_name] = _loaded_workspaces.pop(workspace.name)
    workspace.name = new_name
    return workspace


def get_visible_workspace_names():
    return [key for key in _loaded_workspaces.keys() if key[:2] != '__']


def get_workspace_name(workspace):
    """Returns the name of a workspace given the workspace handle"""
    if isinstance(workspace, str):
        return workspace
    return workspace.name


def delete_workspace(workspace):
    workspace = get_workspace_handle(workspace)
    remove_workspace(workspace)
    del workspace


def workspace_exists(ws_name):
    return ws_name in _loaded_workspaces
