"""
Mantid MSlice
=============

A PyQt-based version of the MSlice (http://mslice.isis.rl.ac.uk) program based
on Mantid (http://www.mantidproject.org).
"""
from __future__ import print_function

import os
from distutils.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py
from setuptools import find_packages, setup
from subprocess import check_call
import sys

# ==============================================================================
# Constants
# ==============================================================================
NAME = 'mslice'
from mslice import __project_url__, __version__

# ==============================================================================
# Custom distutils commands
# ==============================================================================
class build_qt(_build_py):
    description = "build every qt related resources (.uic and .qrc and .pyc)"

    PACKAGE = NAME

    def finalize_options(self):
        _build_py.finalize_options(self)
        self.packages = [NAME]

    def compile_src(self, src, dest):
        compiler = self.get_compiler(src)
        if not compiler:
            return
        dir = os.path.dirname(dest)
        self.mkpath(dir)
        sys.stdout.write("compiling %s -> %s\n" % (src, dest))
        try:
            compiler(src, dest)
        except Exception, e:
            sys.stderr.write('[Error] %r\n' % str(e))

    def run(self):
        for dirpath, _, filenames in os.walk(self.get_package_dir(self.PACKAGE)):
            package = dirpath.split(os.sep)
            for filename in filenames:
                module = self.get_module_name(filename)
                module_file = self.get_module_outfile(self.build_lib, package, module)
                src_file = os.path.join(dirpath, filename)
                self.compile_src(src_file, module_file)
        _build_py.run(self)

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


class build(_build):

    # 'sub_commands': a list of commands this command might have to run to
    # get its work done.  See cmd.py for more info.
    sub_commands = [
        ('build_qt', lambda _: True),
        ] + _build.sub_commands

# ==============================================================================
# Setup arguments
# ==============================================================================
setup_args = dict(name=NAME,
                  version=__version__,
                  description='Visualise and slice data from Mantid',
                  author='The Mantid Project',
                  url=__project_url__,
                  keywords='PyQt4,',
                  packages=find_packages(),
                  classifiers=['Operating System :: MacOS',
                               'Operating System :: Microsoft :: Windows',
                               'Operating System :: POSIX :: Linux',
                               'Programming Language :: Python :: 2.7',
                               'Development Status :: 4 - Beta',
                               'Topic :: Scientific/Engineering'],
                  cmdclass={'build_qt': build_qt,
                            'build': build,
                            }
)

# ==============================================================================
# Setuptools deps
#==============================================================================
setup_args['install_requires'] = ['numpy', 'matplotlib >= 1.5']

setup_args['entry_points'] = {
    'gui_scripts': [
        'mslice = mslice.app.main'
    ]
}

#==============================================================================
# Main setup
#==============================================================================
setup(**setup_args)