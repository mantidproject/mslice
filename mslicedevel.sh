#!/bin/bash
#
# Basic wrapper script for running in development mode. It assumes the current
# working directory is the directory containing this script.
#

# ~EXPLICIT VARIABLE DEFINITIONS~
# PLEASE SPECIFY PATHS HERE IF THEY ARE NOT FOUND AUTOMATICALLY
  MAMBA_DIR=
  CONDA_SH_PATH=

set_mamba_dir () {
  if [[ $MAMBA_DIR != "" ]]; then return; fi

  IFS=: read -r -a patharr <<<"$PATH"
  for dir in "${patharr[@]}"; do
      if [[ "$dir" == *"mambaforge"* && "$dir" != *"mnt"* ]]; then
        IFS=/ read -r -a split_dir <<<"$dir"
          for sub_str in "${split_dir[@]}"; do
            MAMBA_DIR=$MAMBA_DIR"/""$sub_str"
            if [[ "$sub_str" == "mambaforge" ]]; then break; fi
          done
      fi
  done
}

set_conda_sh_path () {
  if [[ $CONDA_SH_PATH != "" ]]; then return; fi
  CONDA_SH_PATH=$MAMBA_DIR"/etc/profile.d/conda.sh"
}

check_paths () {
  if ! [ -d $MAMBA_DIR ]; then
    echo "ERROR: THE MAMBA DIRECTORY COULD NOT BE FOUND. PLEASE SPECIFY EXPLICITLY IN mslicedevel.sh"
    exit 1
  fi

  if ! [ -f $CONDA_SH_PATH ]; then
    echo "ERROR: CONDA.SH COULD NOT BE FOUND. PLEASE SPECIFY EXPLICITLY IN mslicedevel.sh"
    exit 1
  fi
}

set_mamba_dir
set_conda_sh_path
check_paths

# Check if mantidnightly needs to be activated first
if [ "$CONDA_DEFAULT_ENV" != "mantidnightly" ]; then
  source $CONDA_SH_PATH
  conda activate mantidnightly
fi

if [ "$CONDA_DEFAULT_ENV" != "mantidnightly" ]; then
  # This allows to start MSlice when there is no conda available or activating mantidnightly failed
  MANTIDPYTHON=/opt/mantidworkbenchnightly/bin/python
else
  MANTIDPYTHON=python
fi

env PYTHONPATH=$(dirname $0)/src:$PYTHONPATH $MANTIDPYTHON scripts/start_mslice.py
