"""Defines the command-line interface to MSlice
"""
import plotting.pyplot as _plt
from plotting.pyplot import *
import _mslice_commmands
from _mslice_commmands import *

# Define names imported by * imports
__all__ = dir(_plt) + dir(_mslice_commmands)
