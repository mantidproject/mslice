

class WorkspaceView(object):
    error_occurred = None

    def __init__(self):
        raise Exception("This abstact base class must not be instantiated")

    def display_loaded_workspaces(self, workspaces):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_workspace_to_load_path(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_workspace_to_save_filepath(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_workspace_new_name(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_workspace_efixed(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_workspace_index(self, ws_name):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_workspace_selected(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def set_workspace_selected(self, index):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_select_only_one_workspace(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_select_one_workspace(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_select_one_or_more_workspaces(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def get_presenter(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def confirm_overwrite_workspace(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def no_workspace_has_been_loaded(self, filename=None):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_unable_to_open_file(self, filename=None):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_invalid_save_path(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def error_unable_to_save(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")

    def clear_displayed_error(self):
        raise NotImplementedError("This method must be implemented in a concrete view before being called")
