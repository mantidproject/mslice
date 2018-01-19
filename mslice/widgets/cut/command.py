"""Enumeration for operations allowed in this module
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division, print_function)

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------

class Command(object):
    Plot = -999
    PlotOver = -998
    SaveToWorkspace = -997
    SaveToAscii = -996
    AxisChanged = -995
    PlotFromWorkspace = -994
    PlotOverFromWorkspace = -993
