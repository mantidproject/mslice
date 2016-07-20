class WorkspaceView(object):
    #@abc.abstractmethod
    def display_loaded_workspaces(self, workspaces):
        pass
    #@abc.abstractmethod
    def get_workspace_to_load_path(self):
        pass
    #@abc.abcabstractmethod
    def get_workspace_to_load_name(self):
        pass
    #@abc.abstractmethond
    def get_workspace_to_save_filepath(self):
        pass
    #@abc.abstractmethod
    def get_workspace_new_name(self):
        pass
    #@abc.abstractmethod
    def get_workspace_selected(self):
        pass
    #@abc.abstractmethod
    def get_workspace_to_load_path(self):
        pass
    #@abc.abcabstractmethod
    def get_workspace_to_load_name(self):
        pass
    #@abc.abstractmethond
    def get_workspace_to_save_filepath(self):
        pass
    #@abc.abstractmethod
    def get_workspace_new_name(self):
        pass

    #@abc.abstractmethod
    def error_select_only_one_workspace(self):
        pass

    #@abc.abstractmethod
    def error_select_one_workspace(self):
        pass

    #@abc.abstractmethod
    def error_select_one_or_more_workspaces(self):
        pass