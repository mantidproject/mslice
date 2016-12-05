::
:: Launch MSlice using the Mantid Python wrappers. It is currently hardcoded to
:: use the default install location
::

set MANTIDPYTHON=C:\MantidInstall\bin\mantidpython.bat
set MANTIDPYTHON_ARGS=--classic
set MAIN_SCRIPT=%~dp0start_mslice.py

:: Run
%MANTIDPYTHON% %MANTIDPYTHON_ARGS% %MAIN_SCRIPT%
