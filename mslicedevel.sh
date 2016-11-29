#!/bin/sh
#
# Basic wrapper script for running in development mode. It assumes the current
# working directory is the directory containing this script.
#
PY_VERS=2.7

python setup.py build
# The lib directory can contain the os and architecture
if [ -d "build/lib" ]; then
  LOCAL_PYTHONPATH=$PWD/build/lib
else
  LOCAL_PYTHONPATH=$PWD/build/lib.linux-$(uname -p)-${PY_VERS}
fi
PYTHONPATH=$LOCAL_PYTHONPATH:$PYTHONPATH build/scripts-${PY_VERS}/mslice
