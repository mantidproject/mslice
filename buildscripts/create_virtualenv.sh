#!/bin/bash
# Create a virtual environment in the given directory
# using the given python executable

python_exe=$1
env_dir=$2

# install virtualenv locally if it is not present
$python_exe -c 'import virtualenv' > /dev/null 2>&1 || $python_exe -m pip install --user virtualenv

# check if the environment already exists
if [ ! -d "$env_dir" ]
then
  # Create the environment
  echo "Creating environment in ${env_dir}..."
  ${python_exe} -m virtualenv --system-site-packages ${env_dir}
else
  echo "Environment ${env_dir} already exists, doing nothing."
fi
