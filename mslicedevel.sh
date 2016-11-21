#!/bin/sh
#
# Basic wrapper script for running in development mode
#
PY_VERS=2.7

python setup.py build
PYTHONPATH=$PWD/build/lib.linux-$(uname -p)-${PY_VERS}:$PYTHONPATH
build/scripts-${PY_VERS}/mslice
