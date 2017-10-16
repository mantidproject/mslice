"""
Mantid MSlice
=============

A PyQt-based version of the MSlice (http://mslice.isis.rl.ac.uk) program based
on Mantid (http://www.mantidproject.org).
"""
from __future__ import print_function

import os
from setuptools.command.build_py import build_py as _build_py
from setuptools import find_packages, setup
from subprocess import check_call
import sys

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


def get_data_files():
    """Return data_files in a platform dependent manner"""
    if sys.platform.startswith('linux'):
        data_files = [('share/applications', ['scripts/mslice.desktop']),
                      ('share/pixmaps', ['resources/images/mslice_logo.png'])]
    else:
        data_files = []
    return data_files


# ==============================================================================
# Custom distutils build & install commands
# ==============================================================================

class build_py(_build_py):
    description = "build pure python + qt related resources (.uic and .qrc and .pyc)"

    user_options = _build_py.user_options + [
        ('inplace', 'i', "build inplace and not to build directory")
    ]

    boolean_options = _build_py.boolean_options + ['inplace']

    PACKAGE = NAME

    def initialize_options(self):
        _build_py.initialize_options(self)
        self.inplace = False

    def finalize_options(self):
        _build_py.finalize_options(self)
        self.distribution.packages.append(NAME)

    def compile_src(self, src, dest):
        compiler = self.get_compiler(src)
        if not compiler:
            return
        dir = os.path.dirname(dest)
        self.mkpath(dir)
        sys.stdout.write("compiling %s -> %s\n" % (src, dest))
        try:
            compiler(src, dest)
        except Exception as e:
            sys.stderr.write('[Error] %r\n' % str(e))

    def run(self):
        from distutils.dep_util import newer
        for dirpath, _, filenames in os.walk(self.get_package_dir(self.PACKAGE)):
            package = dirpath.split(os.sep)
            for filename in filenames:
                src_file, module_file = self.get_inout(package, dirpath, filename)
                if newer(src_file, module_file):
                    self.compile_src(src_file, module_file)
        _build_py.run(self)

    def get_outputs(self, include_bytecode=1):
        '''Return the list of outputs that would be generated
           if this command were run
        '''
        outputs = _build_py.get_outputs(self, include_bytecode)
        for dirpath, _, filenames in os.walk(self.get_package_dir(self.PACKAGE)):
            package = dirpath.split(os.sep)
            for filename in filenames:
                _, module_file = self.get_inout(package, dirpath, filename)
                outputs.append(module_file)
            if include_bytecode:
                if self.compile:
                    outputs.append(module_file + "c")
                if self.optimize > 0:
                    outputs.append(module_file + "o")

        return outputs

    def get_inout(self, package, dirpath, filename):
        src_file = os.path.join(dirpath, filename)
        module_name = self.get_module_name(filename)
        if self.inplace:
            if self.is_compilation_required(src_file):
                module_file = os.path.join(dirpath, module_name) + '.py'
            else:
                module_file = src_file
        else:
            module_file = self.get_module_outfile(self.build_lib, package, module_name)

        return src_file, module_file

    def is_compilation_required(self, source_file):
        '''Is something above a simply copy required'''
        return self.get_compiler(source_file) is not None

    @staticmethod
    def compile_ui(ui_file, py_file):
        from PyQt4 import uic

        with open(py_file, 'w') as fp:
            uic.compileUi(ui_file, fp)

    @staticmethod
    def compile_qrc(qrc_file, py_file):
        check_call(['pyrcc4', qrc_file, '-o', py_file])

    def get_compiler(self, source_file):
        name = 'compile_' + source_file.rsplit(os.extsep, 1)[-1]
        return getattr(self, name, None)

    @staticmethod
    def get_module_name(src_filename):
        name, ext = os.path.splitext(src_filename)
        return {'.qrc': '%s_rc', '.ui': '%s_ui'}.get(ext, '%s') % name


# ==============================================================================
# Setup arguments
# ==============================================================================
setup_args = dict(name=NAME,
                  version=__version__,
                  description='Visualise and slice data from Mantid',
                  author='The Mantid Project',
                  author_email='mantid-help@mantidproject.org',
                  url=__project_url__,
                  keywords=['PyQt4'],
                  packages=find_packages(exclude=["misc"]),
                  data_files=get_data_files(),
                  # Fool setup.py to running the tests on a built copy (this feels like a hack)
                  use_2to3=True,
                  # Install this as a directory
                  zip_safe=False,
                  classifiers=['Operating System :: MacOS',
                               'Operating System :: Microsoft :: Windows',
                               'Operating System :: POSIX :: Linux',
                               'Programming Language :: Python :: 2.7',
                               'Development Status :: 4 - Beta',
                               'Topic :: Scientific/Engineering'],
                  cmdclass={'build_py': build_py})

# ==============================================================================
# Setuptools deps
# ==============================================================================
# Running setup command requires the following dependencies
setup_args['setup_requires'] = read_requirements_from_file(os.path.join(THIS_DIR, 'setup-requirements.txt'))

# User installation requires the following dependencies
# PyQt4 cannot be installed from pip so they cannot be added here
install_requires = setup_args['install_requires'] = \
    read_requirements_from_file(os.path.join(THIS_DIR, 'install-requirements.txt'))
# Testing requires
setup_args['tests_require'] = read_requirements_from_file(os.path.join(THIS_DIR, 'test-requirements.txt')) \
    + install_requires

# Startup scripts - these use the mantidpython wrappers so we cannot
# go through the entry_points mechanism
scripts = ['scripts/start_mslice.py']
if os.name == 'nt':
    scripts.append('scripts/mslice.bat')
else:
    scripts.append('scripts/mslice')
setup_args['scripts'] = scripts

# ==============================================================================
# Main setup
# ==============================================================================
setup(**setup_args)
