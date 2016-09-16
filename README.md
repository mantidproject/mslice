# MSlice3.0
[![Build Status](https://travis-ci.org/mantidproject/mslice.svg?branch=restructured_code)] (https://travis-ci.org/mantidproject/mslice/)[![Coverage Status](https://coveralls.io/repos/github/mantidproject/mslice/badge.svg?branch=restructured_code)](https://coveralls.io/github/mantidproject/mslice?branch=restructured_code)

##Running mslice

###Windows
Run the `start_mslice.bat` in the root directory of the project and the MSlice GUI and the accompanying console should open.
###Linux & Mac 
Run the `start_mslice.sh` in the root directory of the project and the MSlice GUI and the accompanying console should open

##If you have installed Mantid to a custom directory
###Windows
It assumes you have a nightly build of Mantid at `C:\MantidInstall\` . If you have a nightly build installed somewhere else on your computer then open the `start_mslice.bat` in a text editor. 
It should look like
```
C:\MantidInstall\bin\mantidpython.bat --matplotlib=qt -i start_script.ipy
```
Replace the `C:\MantidInstall\` with the root directory of your Mantid installation. After saving the file running again 
###Linux 
It assumes you have a nightly build of Mantid at `/opt/mantidnightly/` . If you have a nightly build installed somewhere else on your computer then open the `start_mslice.sh` in a text editor. 
It should look like
```
/opt/mantidnightly/bin/mantidpython --matplotlib -i start_script.ipy
```
Replace the `/opt/mantidnightly/` with the root directory of your Mantid installation. After saving the file running again 

##Note to developers
If you are running a build of Mantid complied on your machine. The actual location of the mantidpython interpreter will depend upon the compiler used to build mantid and the compilation options. To get MSlice running you will have to manually search for it and put in the MSlice start_up script
###Windows
To find your `mantidpython.bat` file open your build directory in and search for `mantidpython.bat`. After locating the `mantidpython.bat` open the `start_mslice.bat` in a text editor and replace 
```
C:\MantidInstall\bin\mantidpython.bat --matplotlib -i start_script.ipy
```
with 
```
<path_to_mantidpython.bat> --matplotlib -i start_script.ipy
```
###Linux
To find your `mantidpython` executable file open your build directory in and search for `mantidpython`. After locating the `mantidpython` open the `start_mslice.sh` in a text editor and replace 
```
/opt/mantidnightly/bin/mantidpython --matplotlib -i start_script.ipy
```
with 
```
<path_to_mantidpython> --matplotlib -i start_script.ipy
```
