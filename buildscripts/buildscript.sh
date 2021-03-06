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
    py3)
        py_exe=/usr/bin/python3
        pkg_name=mantidnightly
    ;;
    *)
        echo "Unknown python version requested '$py_ver'"
        exit 1
esac
# X server
XVFB_SERVER_NUM=101
# Qt4 backends
export QT_API=pyqt5
export MPLBACKEND=Qt5Agg

function onexit {
  deactivate
  rm -fr ${venv_dir}
  sudo dpkg --purge ${pkg_name}
}
trap onexit EXIT

# ------------------------------------------------------------------------------
# terminate existing Xvfb sessions
# ------------------------------------------------------------------------------
if [ $(command -v xvfb-run) ]; then
    echo "Terminating existing Xvfb sessions"

    # Kill Xvfb processes
    killall Xvfb || true

    # Remove Xvfb X server lock files
    rm -f /tmp/.X${XVFB_SERVER_NUM}-lock
fi


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
xvfb-run --server-args="-screen 0 640x480x24" \
  --server-num=${XVFB_SERVER_NUM} python setup.py nosetests
