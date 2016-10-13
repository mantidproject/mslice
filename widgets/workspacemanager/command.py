"""Defines enumerated values for operations available in the workspace manager.
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class Command(object):
    SaveSelectedWorkspace = 1
    RemoveSelectedWorkspaces = 2
    LoadWorkspace = 3
    ComposeWorkspace = 5  # On hold for now
    RenameWorkspace = 1000
    SelectionChanged = -1799
