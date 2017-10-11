from __future__ import (absolute_import, division, print_function)
"""Defines the interface to an object responsible for performaing operations on
workspaces
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import abc

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class WorkspaceProvider(object):

    @abc.abstractmethod
    def get_workspace_names(self):
        pass

    @abc.abstractmethod
    def delete_workspace(self, workspace):
        pass

    @abc.abstractmethod
    def load(self, filename, output_workspace):
        pass

    @abc.abstractmethod
    def rename_workspace(self, selected_workspace, new_name):
        pass

    @abc.abstractmethod
    def save_nexus(self, workspace, path):
        pass

    def get_workspace_handle(self, workspace_name):
        pass

    def get_workspace_name(self, workspace_handle):
        pass

    def getComment(self, workspace):
        pass
