#!/bin/sh
#
# Basic wrapper script for running in development mode. It assumes the current
# working directory is the directory containing this script.
#
python setup.py build_py --inplace
sh scripts/mslice
