from __future__ import (absolute_import, division, print_function)


class CLIDataLoaderWidget:

    def __init__(self):
        pass

    def show_busy(self, is_busy):
        pass

    def keyPressEvent(self, event):
        pass

    def activated(self, file_clicked):
        pass

    def enter_dir(self, directory):
        pass

    def refresh(self):
        pass

    def _update_from_path(self):
        pass

    def back(self):
        pass

    def load(self, merge):
        pass

    def sort_files(self, column):
        pass

    def go_to_home(self):
        pass

    def validate_selection(self):
        pass

    def get_selected_file_paths(self):
        pass

    def get_workspace_efixed(self, workspace, hasMultipleWS=False, default_value=None):
        pass

    def get_presenter(self):
        pass

    def error_unable_to_open_file(self, filename=None):
        pass

    def error_merge_different_file_formats(self):
        pass

    def no_workspace_has_been_loaded(self, filename=None):
        pass

    def confirm_overwrite_workspace(self):
        pass

    def error_loading_workspace(self, message):
        pass

    def _display_error(self, error_string):
        pass

    def _clear_displayed_error(self):
        pass
