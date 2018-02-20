#!/bin/bash
# Install a nightly build from Jenkins
#
set -e
JENKINS_ROOT_URL=http://builds.mantidproject.org
NIGHTLY_PY2_JOB=master_clean-ubuntu
NIGHTLY_PY3_JOB=master_clean-ubuntu-python3

function download_latest_build() {
  local job_url=$1
  local outpath=$2
  local last_nightly_url=$(curl -sL ${job_url}/api/json | jq '.lastSuccessfulBuild.url' | cut -d'"' -f 2)
  # -1 will only work if it is the last file
  local artefact_path=$(curl -sL ${last_nightly_url}/api/json | jq '.artifacts[-1].relativePath' | cut -d'"' -f2)

  echo "Downloading nightly from ${last_nightly_url}/artifact/${artefact_path}"
  curl -L ${last_nightly_url}/artifact/${artefact_path} -o $outpath
}

if [ $# != 1 ]; then
  echo "Usage: install_mantidnightly_sh 2.7|3.4"
  exit 1
fi

# Add mantid dependencies repo
sudo apt-add-repository ppa:mantid/mantid -y && sudo apt-get update -qq

# Install helper utilities
sudo apt-get install -y gdebi-core jq

# Download nightly build for the correct python version
if [ "$1" = "2.7" ]; then
  nightly_job=${NIGHTLY_PY2_JOB}
elif [ "$1" = "3.4" ]; then
  nightly_job=${NIGHTLY_PY3_JOB}
else
  echo "Unknown python version. Argument should be on of 2.7|3.4"
  exit 1
fi
nightly_deb=/tmp/mantidnightly.deb
download_latest_build ${JENKINS_ROOT_URL}/job/${nightly_job} ${nightly_deb}

# Install
sudo gdebi --option=APT::Get::force-yes=1,APT::Get::Assume-Yes=1 -n  ${nightly_deb}

# Install X virtual framebuffer to emulate X server
sudo apt-get install -y xvfb

# Configure mantid options
mantid_cfg_dir=$HOME/.mantid
mkdir -p ${mantid_cfg_dir} && echo -e "UpdateInstrumentDefinitions.OnStartup=0\nCheckMantidVersion.OnStartup=0\n" > ${mantid_cfg_dir}/Mantid.user.properties
