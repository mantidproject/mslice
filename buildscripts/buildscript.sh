#!/bin/bash
# Run the build.
# User supplies the version of python to run.sh with and
# a directory that should contain a mantid debian file
set -ex

script_dir=$(cd "$(dirname "$0")"; pwd -P)

py_ver=$1
mantid_deb_dir=$2

venv_dir=$PWD/venv-${py_ver}

# determine python executable
case "$py_ver" in
    py2)
        py_exe=/usr/bin/python
        pkg_name=mantidnightly
    ;;
    py3)
        py_exe=/usr/bin/python3
        pkg_name=mantidnightly-python3
    ;;
    *)
        echo "Unknown python version requested '$py_ver'"
        exit 1
esac
# Qt4 backends
export QT_API=pyqt
export MPLBACKEND=Qt4Agg

function onexit {
  deactivate
  rm -fr ${venv_dir}
  sudo dpkg --purge ${pkg_name}
}
trap onexit EXIT

# ------------------------------------------------------------------------------
# pre-installation
# ------------------------------------------------------------------------------
# find mantid package to install
if [ ! -d "$mantid_deb_dir" ]; then
    echo "'$mantid_deb_dir' is not a directory"
    exit 1
fi

if [ "$(ls -1 ${mantid_deb_dir}/*.deb | wc -l)" != 1 ]; then
    echo "Found more than 1 mantid package in $mantid_deb_dir"
    exit 1
fi
echo "Install latest mantid build"
# assume suoders allows us to do this
sudo gdebi -n $(find "${mantid_deb_dir}" -name '*.deb' -type f -print)

echo "Configuring mantid properties"
userprops=~/.mantid/Mantid.user.properties
rm -f $userprops
# Turn off any auto updating on startup
echo "UpdateInstrumentDefinitions.OnStartup = 0" > $userprops
echo "usagereports.enabled = 0" >> $userprops
echo "CheckMantidVersion.OnStartup = 0" >> $userprops

# use a virtual environment
${script_dir}/create_virtualenv.sh ${py_exe} ${venv_dir}
source ${venv_dir}/bin/activate
# ensure mantid is on the python path
export PYTHONPATH=/opt/${pkg_name}/bin:/opt/${pkg_name}/lib

# ------------------------------------------------------------------------------
# installation
# ------------------------------------------------------------------------------
# install mslice requirements
pip install -r setup-requirements.txt -r install-requirements.txt -r test-requirements.txt

# ------------------------------------------------------------------------------
# test steps
# ------------------------------------------------------------------------------
# flake 8
python setup.py flake8

# test
python setup.py nosetests
