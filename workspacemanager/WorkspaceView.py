from workspacemanager.WorkspaceManagerPresenter import WorkspaceManagerPresenter
from workspacemanager.WorkspaceProvider import WorkspaceProvider


class WorkspaceView(object):
    def __init__(self):
        workspace_provider = WorkspaceProvider()
        self._presenter = WorkspaceManagerPresenter(self,workspace_provider)
        print 'wsmViewAbstract created',id(self)
    def display_loaded_workspaces(self, workspaces):
        pass

    def get_workspace_to_load_path(self):
        pass

    def get_workspace_to_load_name(self):
        pass

    def get_workspace_to_save_filepath(self):
        pass

    def get_workspace_new_name(self):
        pass

    def get_workspace_selected(self):
        pass

    def get_workspace_to_load_path(self):
        pass

    def get_workspace_to_load_name(self):
        pass

    def get_workspace_to_save_filepath(self):
        pass

    def get_workspace_new_name(self):
        pass

    def error_select_only_one_workspace(self):
        pass

    def error_select_one_workspace(self):
        pass

    def error_select_one_or_more_workspaces(self):
        pass

    def get_presenter(self):
        return self._presenter