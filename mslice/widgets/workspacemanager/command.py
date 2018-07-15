"""Defines enumerated values for operations available in the workspace manager.
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------


class Command(object):
    SaveSelectedWorkspaceNexus = 1
    RemoveSelectedWorkspaces = 2
    LoadWorkspace = 3
    Subtract = 4
    ComposeWorkspace = 5  # On hold for now
    Add = 6
    SaveSelectedWorkspaceAscii = 7
    SaveSelectedWorkspaceMatlab = 8
    SaveToADS = 9
    RenameWorkspace = 1000
    CombineWorkspace = 1010
    SelectionChanged = -1799
