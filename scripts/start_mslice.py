#!/usr/bin/python
from __future__ import absolute_import

from mslice.app import startup

if __name__ == '__main__':
    try: # check if started from within mantidplot
        import mantidplot  # noqa
        startup(with_ipython=False)
    except ImportError:
        startup(with_ipython=True)
