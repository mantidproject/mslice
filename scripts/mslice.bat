::
:: Launch MSlice using the Mantid Python wrappers. Uses MANTIDPATH
:: to determine where mantid is located otherwise it defaults to a
:: C:\MantidInstall\bin
::


set MAIN_SCRIPT=%~dp0start_mslice.py

if DEFINED CONDA_PREFIX (
  python %MAIN_SCRIPT%
) else if DEFINED CONDA_DEFAULT_ENV (
  python %MAIN_SCRIPT%
) else if DEFINED MANTIDPATH (
  set MANTIDPYTHON=%MANTIDPATH%\mantidpython.bat
) else (
  set MANTIDPYTHON=C:\MantidInstall\bin\mantidpython.bat
)

set MANTIDPYTHON_ARGS=--classic

:: Run
%MANTIDPYTHON% %MANTIDPYTHON_ARGS% %MAIN_SCRIPT%
