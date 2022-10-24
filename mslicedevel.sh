#!/bin/bash
#
# Basic wrapper script for running in development mode. It assumes the current
# working directory is the directory containing this script.
#

# Check if mantidnightly needs to be activated first
if [ "$CONDA_DEFAULT_ENV" != "mantidnightly" ]; then
  source /opt/mambaforge/etc/profile.d/conda.sh
  conda activate mantidnightly
fi

if [ "$CONDA_DEFAULT_ENV" != "mantidnightly" ]; then
  # This allows to start MSlice when there is no conda available or activating mantidnightly failed
  MANTIDPYTHON=/opt/mantidworkbenchnightly/bin/python
else
  MANTIDPYTHON=python
fi

env PYTHONPATH=$(dirname $0):$PYTHONPATH $MANTIDPYTHON scripts/start_mslice.py
