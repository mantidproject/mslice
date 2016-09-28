"""Defines the command-line interface to MSlice
"""

__all__ = []
import plotting.pyplot as _plt
from plotting.pyplot import *
__all__ += dir(_plt)

import _mslice_commmands
from _mslice_commmands import *
__all__ += dir(_mslice_commmands)