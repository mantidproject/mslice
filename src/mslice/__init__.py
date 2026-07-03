"""
Mantid MSlice
=============

A PyQt-based version of the MSlice (http://mslice.isis.rl.ac.uk) program based
on Mantid (http://www.mantidproject.org).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mslice")
except PackageNotFoundError:
    __version__ = "0+unknown"

__project_url__ = "https://github.com/mantidproject/mslice"
