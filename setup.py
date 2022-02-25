"""
Mantid MSlice
=============

A PyQt-based version of the MSlice (http://mslice.isis.rl.ac.uk) program based
on Mantid (http://www.mantidproject.org).
"""
from __future__ import print_function

import fnmatch
import os
import sys

from setuptools import find_packages, setup

from mslice import __project_url__, __version__

# ==============================================================================
# Constants
# ==============================================================================
NAME = 'mslice'
THIS_DIR = os.path.dirname(__file__)


# ==============================================================================
# Package requirements helper
# ==============================================================================

def read_requirements_from_file(filepath):
    '''Read a list of requirements from the given file and split into a
    list of strings. It is assumed that the file is a flat
    list with one requirement per line.
    :param filepath: Path to the file to read
    :return: A list of strings containing the requirements
    '''
    with open(filepath, 'rU') as req_file:
        return req_file.readlines()

def get_package_data():
    """Return data_files in a platform dependent manner"""
    package_data = []
    package_dir = os.path.join(THIS_DIR, NAME)
    for root, dirnames, filenames in os.walk(package_dir):
        for filename in fnmatch.filter(filenames, '*.ui'):
            package_data.append(os.path.relpath(os.path.join(root, filename), start=package_dir))
    return {NAME: package_data}

def get_data_files():
    """Return data_files in a platform dependent manner"""
    if sys.platform.startswith('linux'):
        data_files = [('share/applications', ['scripts/mslice.desktop']),
                      ('share/pixmaps', ['resources/images/mslice_logo.png'])]
    else:
        data_files = []
    return data_files

# ==============================================================================
# Setup arguments
# ==============================================================================
setup_args = dict(name=NAME,
                  version=__version__,
                  description='Visualise and slice data from Mantid',
                  author='The Mantid Project',
                  author_email='mantid-help@mantidproject.org',
                  url=__project_url__,
                  keywords=['PyQt5'],
                  packages=find_packages(exclude=["misc"]),
                  package_data=get_package_data(),
                  data_files=get_data_files(),
                  entry_points={"console_scripts" : ["mslice = mslice.app.__init__:main"]},
                  # Install this as a directory
                  zip_safe=False,
                  classifiers=['Operating System :: MacOS',
                               'Operating System :: Microsoft :: Windows',
                               'Operating System :: POSIX :: Linux',
                               'Programming Language :: Python :: 3.8',
                               'Development Status :: 4 - Beta',
                               'Topic :: Scientific/Engineering'])

# ==============================================================================
# Setuptools deps
# ==============================================================================
# Running setup command requires the following dependencies
setup_args['setup_requires'] = read_requirements_from_file(os.path.join(THIS_DIR, 'setup-requirements.txt'))

# User installation requires the following dependencies
install_requires = setup_args['install_requires'] = \
    read_requirements_from_file(os.path.join(THIS_DIR, 'install-requirements.txt'))
# Testing requires
setup_args['tests_require'] = read_requirements_from_file(os.path.join(THIS_DIR, 'test-requirements.txt')) \
    + install_requires

# Startup scripts - these use the mantidpython wrappers so we cannot
# go through the entry_points mechanism
#scripts = ['scripts/start_mslice.py']
#if os.name == 'nt':
#    scripts.append('scripts/mslice.bat')
#else:
#    scripts.append('scripts/mslice')
#setup_args['scripts'] = scripts

# ==============================================================================
# Main setup
# ==============================================================================
setup(**setup_args)
