::
:: Launch MSlice using the Mantid Python wrappers. Uses MANTIDPATH
:: to determine where mantid is located otherwise it defaults to a
:: C:\MantidInstall\bin
::

if DEFINED MANTIDPATH (
  set MANTIDPYTHON=%MANTIDPATH%\mantidpython.bat
) else (
  if DEFINED CONDA_PREFIX (
    set MANTIDPYTHON=mantidpython
  ) else (
    set MANTIDPYTHON=C:\MantidInstall\bin\mantidpython.bat
  )
)
set MANTIDPYTHON_ARGS=--classic
set MAIN_SCRIPT=%~dp0start_mslice.py

:: Run
%MANTIDPYTHON% %MANTIDPYTHON_ARGS% %MAIN_SCRIPT%
