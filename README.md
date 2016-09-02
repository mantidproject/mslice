# MSlice3.0
[![Build Status](https://travis-ci.org/mantidproject/mslice.svg?branch=master)] (https://travis-ci.org/mantidproject/mslice/)[![Coverage Status](https://coveralls.io/repos/github/mantidproject/mslice/badge.svg?branch=master)](https://coveralls.io/github/mantidproject/mslice?branch=master)
# MSlice
MSlice is a tool for visualizing cuts and slices from powder diffraction samples. MSlice uses the [Mantid Framework](http://www.mantidproject.org/) for all data manipulations and uses [matplotlib](http://matplotlib.org/) for visualising data

## Installing MSlice
###Requirements
- A recent build of `Mantid` (nightly or recent build of master) 
To get the latest version of Mantid follow [this link](http://download.mantidproject.org/) and click on the download button the header *Nightly Build*
- Matplotlib >=`1.5.1`
Recent versions of Mantid (including the current stable release and beyond) for windows currently ship with a recent version of `matplotlib`. However Mantid for linux comes with a slighlty older version.
To check the version of Matplotlib on your system open the `MantidPlot` application and in the script interpreter at the bottom of the window execute ` import matplotlib; print matplotlib.__version__`
The output should look like
```
In [1]: import matplotlib; print matplotlib.__version__
1.5.1

In [2]:
```
If the version that appears in the interpreter is less that  `1.5.1` then you must upgrade it by executing the following command in the MantidPlot script interpreter
 `%run -m easy_install -- --user "matplotlib>=1.5.1"`
 
 ####Warning
 Executing  `%run -m easy_install -- --user "matplotlib>=1.5.1"` on windows will not and may crash your PC!
 