# Mantid MSlice

## Build status
[![Build Status](https://travis-ci.org/mantidproject/mslice.svg?branch=master)](https://travis-ci.org/mantidproject/mslice/)
[![Coverage Status](https://coveralls.io/repos/github/mantidproject/mslice/badge.svg?branch=master)](https://coveralls.io/github/mantidproject/mslice?branch=master)

## Overview

MSlice is a tool for performing slices and cuts of multi-dimensional data produced by
[Mantid](http://www.mantidproject.org).

## Documentation

The current MSlice documentation can be viewed at

http://mantidproject.github.io/mslice

## Development

The following setup steps are required regardless of the environment:

* install mantid from either http://download.mantidproject.org or yum/apt repositories (nightly on Linux)
* clone this repository

### Command Line

To develop purely on the command line then simply use your favourite editor and run either

* `mslicedevel.bat` (Windows) or
* `./mslicedevel` (Linux)

### PyCharm

Mantid must be installed prior to running this setup.

To set up the [PyCharm IDE](https://www.jetbrains.com/pycharm/) first open PyCharm and select `File->Open Project`. Select the cloned `mslice` directory and select open.
The project layout should be displayed. The first run may take some time to open while PyCharm parses the structure.

You will also need to edit the run configurations:  

![example pycharm run configuration](resources/images/pycharm_run_config.png)

- The startup script is `start_mslice.py`.
- The `bin` directory of an installed version of Mantid must be on the `PATH`.
- Set the environment variable `QT_QPA_PLATFORM_PLUGIN_PATH` to the directory with the QT plugins from the Mantid installation `MantidInstall\plugins\qt5`.
- If you're developing on Windows, the Python interpreter used must be the one shipped with the Mantid installation `MantidInstall/bin/python.exe`.
- If you're developing on Ubuntu, set the Python Interpreter path to `/usr/bin/python3.6`

You can now also develop MSlice using a Mantid conda environment.
First install Mantid using `conda create -n mantidnightly -c conda-forge -c mantid/label/nightly mantid mantidqt qtconsole nose coverage mock`
then add this interpreter by going to the `File->Settings` in PyCharm, then `Project: mslice -> Python Interpreter`,
click the cog on the right side to add an existing interpreter and select `Conda` and `Python 3.8 (mantidnightly)`.
Then go to `Run -> Edit Configurations` and create new configuration with this interpreter.
Specify `start_mslice.py` as the startup script as above, but you do not need to set any environment variable.
To run tests, create a `Nosetests` configuration and specify the `Target` as `Custom` with `mslice.tests`
with the working directory being the mslice package folder (e.g. `<mslice_root>/mslice`).

